import os
import pickle
import warnings
from astropy.io import fits
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from ratansunpy.client import ARClient
from ratansunpy.time import TimeRange

from src.dataset import FITS_Dataset
from src.logger import get_logger
from src.models import ConvAE
from src.plot import visualize_samples, plot_spectral_comparison
from src.utils import train_model, plot_losses, gather_fits_files
from src.constants import (
    PLOTS_DIR,
    AE_OUTPUTS,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)

warnings.simplefilter("ignore", FutureWarning)
logger = get_logger(__file__)


def download_and_prepare_data(config):
    """
    Скачивает, предобрабатывает и сохраняет FITS-файлы для обучения автоэнкодера.

    Эта функция выполняет следующие шаги:
    1. Скачивает FITS-файлы за указанные годы, если это задано в конфиге.
    2. Обрабатывает каждый FITS-файл: обрезает, нормализует и сохраняет в один
       большой numpy-массив.
    3. Сохраняет оригинальные данные для последующей визуализации.
    4. Сохраняет список путей к FITS-файлам, чтобы сохранить соответствие с данными.

    Аргументы:
        config (AEConfig): Объект конфигурации, загруженный из YAML-файла.
    """
    logger.info("Шаг 1: Подготовка данных.")

    ar_fits_dir = os.path.join(RAW_DATA_DIR, config.data.folder_name)
    preprocess_fits_save_path = os.path.join(
        INTERIM_DATA_DIR, config.data.preprocess_fits_name
    )
    original_data_for_viz_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.original_data_for_viz_path
    )
    visualization_filenames_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.visualization_filenames_path
    )
    filepaths_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.filepaths_path
    )

    years = config.data.years
    target_height = config.data.height
    target_width = config.data.width
    target_channels = config.data.channels
    clip_factor = config.data.clip_factor

    if config.data.download_fits:
        logger.info("Скачиваем изначальные данные...")
        ar_timerange = TimeRange(f'{years[0]}-01-01', f'{years[-1]}-01-01')
        ar_client = ARClient()
        ar_client.acquire_data(timerange=ar_timerange)
        filepaths = ar_client.download_data(
            timerange=ar_timerange, save_to=ar_fits_dir
        )
    else:
        logger.info("Собираем список путей до локальный данных...")
        filepaths = gather_fits_files(ar_fits_dir, years)
    
    logger.info(f"Найдено файлов до фильтрации: {len(filepaths)}")

    if not filepaths:
        logger.critical("Не найдено файлов FITS. Завершение.")
        exit()

    with open(filepaths_path, 'wb') as f:
        pickle.dump(filepaths, f)
    logger.info(f"Список путей к FITS-файлам сохранен в: {filepaths_path}")

    if config.data.preprocess_fits or not os.path.exists(
        preprocess_fits_save_path
    ):
        logger.info("Создаём и сохраняем датасет...")
        
        all_i_raw, all_v_raw = [], []
        processed_filepaths = []
        dropped_count = 0
        trimmed_count = 0

        logger.info(
            f"Обработка и обрезка данных до размера ({target_channels}, "
            f"{target_height}, {target_width})..."
        )
        for filepath in tqdm(filepaths, desc="Обработка FITS файлов"):
            try:
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    current_channels, current_height, current_width = data.shape
                    
                    if (
                        current_channels < 2
                        or current_height < target_height
                        or current_width != target_width
                    ):
                        logger.debug(f"  Пропускаем {filepath}: несоответствие размеров/каналов.")
                        dropped_count += 1
                        continue
                    
                    i_data_cropped = data[0, :target_height, :]
                    v_data_cropped = data[1, :target_height, :]
                    
                    if np.isnan(i_data_cropped).any():
                        i_data_cropped = np.nan_to_num(i_data_cropped, nan=0.0)
                    if np.isnan(v_data_cropped).any():
                        v_data_cropped = np.nan_to_num(v_data_cropped, nan=0.0)
                        
                    if current_height > target_height:
                        trimmed_count += 1
                        
                    all_i_raw.append(i_data_cropped)
                    all_v_raw.append(v_data_cropped)
                    processed_filepaths.append(filepath)
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {filepath}: {e}")
                dropped_count += 1
                continue
        
        filepaths = processed_filepaths
        logger.info(f"Файлов после фильтрации и обрезки: {len(filepaths)}")
        logger.info(f"Файлов пропущено (размеры/каналы/ошибка): {dropped_count}")
        logger.info(f"Файлов обрезано по высоте: {trimmed_count}")
        
        if not filepaths:
            logger.critical("Нет файлов для обработки. Завершение.")
            exit()
        
        original_data_for_viz_np = np.stack(
            [np.array(all_i_raw), np.array(all_v_raw)], axis=1
        ).astype(np.float32)
        np.save(original_data_for_viz_path, original_data_for_viz_np)
        logger.info(
            f"Оригинальные данные для визуализации сохранены в: {original_data_for_viz_path}"
        )

        all_i_raw_np = np.array(all_i_raw, dtype=np.float64)
        all_v_raw_np = np.array(all_v_raw, dtype=np.float64)
        
        logger.info(f"Shape of all_I_raw_np: {all_i_raw_np.shape}")
        logger.info(f"Number of NaNs in all_I_raw_np: {np.isnan(all_i_raw_np).sum()}")
        logger.info(f"Min value in all_I_raw_np: {np.nanmin(all_i_raw_np):.5f}")
        logger.info(f"Max value in all_I_raw_np: {np.nanmax(all_i_raw_np):.5f}")
        
        logger.info(f"Shape of all_V_raw_np: {all_v_raw_np.shape}")
        logger.info(f"Number of NaNs in all_V_raw_np: {np.isnan(all_v_raw_np).sum()}")
        logger.info(f"Min value in all_V_raw_np: {np.nanmin(all_v_raw_np):.5f}")
        logger.info(f"Max value in all_V_raw_np: {np.nanmax(all_v_raw_np):.5f}")
        
        global_median_i = np.nanmedian(all_i_raw_np)
        global_mad_i = np.nanmedian(np.abs(all_i_raw_np - global_median_i))
        global_median_v = np.nanmedian(all_v_raw_np)
        global_mad_v = np.nanmedian(np.abs(all_v_raw_np - global_median_v))
        logger.info(f"DEBUG (Pre-clip Stats) - I: median={global_median_i:.5f}, MAD={global_mad_i:.5f}")
        logger.info(f"DEBUG (Pre-clip Stats) - V: median={global_median_v:.5f}, MAD={global_mad_v:.5f}")
        
        all_i_clipped_np = np.clip(
            all_i_raw_np,
            global_median_i - clip_factor * global_mad_i,
            global_median_i + clip_factor * global_mad_i,
        )
        all_v_clipped_np = np.clip(
            all_v_raw_np,
            global_median_v - clip_factor * global_mad_v,
            global_median_v + clip_factor * global_mad_v,
        )
        
        logger.info(f"I channel clipped. New Min: {np.min(all_i_clipped_np):.5f}, New Max: {np.max(all_i_clipped_np):.5f}")
        logger.info(f"V channel clipped. New Min: {np.min(all_v_clipped_np):.5f}, New Max: {np.max(all_v_clipped_np):.5f}")

        final_median_i = np.nanmedian(all_i_clipped_np)
        final_mad_i = np.nanmedian(np.abs(all_i_clipped_np - final_median_i))
        final_median_v = np.nanmedian(all_v_clipped_np)
        final_mad_v = np.nanmedian(np.abs(all_v_clipped_np - final_median_v))

        final_mad_i = max(final_mad_i, 1e-8)
        final_mad_v = max(final_mad_v, 1e-8)
        
        logger.info(f"Глобальная статистика (после клиппинга) - I: median={final_median_i:.5f}, MAD={final_mad_i:.5f}")
        logger.info(f"Глобальная статистика (после клиппинга) - V: median={final_median_v:.5f}, MAD={final_mad_v:.5f}")
        
        normalized_i = (all_i_clipped_np - final_median_i) / final_mad_i
        normalized_v = (all_v_clipped_np - final_median_v) / final_mad_v
        
        normalized_data_np = np.stack(
            [normalized_i, normalized_v], axis=1
        ).astype(np.float32)
        
        logger.info(f"Конечное количество нормализованных примеров: {len(filepaths)}")
        logger.info(f"Шейп нормализованных данных (для модели): {normalized_data_np.shape}")
        
        np.save(preprocess_fits_save_path, normalized_data_np)
        logger.info(f"Нормализованные данные сохранены в: {preprocess_fits_save_path}")
    else:
        logger.info("Данные уже предобработаны. Шаг пропущен.")

    logger.info("Выбираем и сохраняем имена файлов для визуализации...")
    num_visual_samples = min(config.output.num_visual_examples, len(filepaths))
    
    if num_visual_samples > 0:
        if os.path.exists(visualization_filenames_path):
            os.remove(visualization_filenames_path)
            logger.info(f"Удален старый файл: {visualization_filenames_path}")
            
        selected_viz_filepaths = np.random.choice(
            filepaths, num_visual_samples, replace=False
        )
        with open(visualization_filenames_path, 'w') as f:
            for item in selected_viz_filepaths:
                f.write(f"{item}\n")
        logger.info(
            f"Выбрано и сохранено {num_visual_samples} имен файлов для визуализации в: "
            f"{visualization_filenames_path}"
        )
    else:
        logger.info(
            "Количество семплов для визуализации равно 0. Шаг пропущен."
        )

def train_ae_model(config):
    """
    Обучает автоэнкодер на предобработанных данных или загружает обученную модель.

    Функция выполняет следующие шаги:
    1. Загружает предобработанные данные.
    2. Инициализирует датасеты и загрузчики данных для обучения и валидации.
    3. Если `train_model` True, запускает цикл обучения, сохраняет модель и графики потерь.
    4. Если `train_model` False, загружает уже обученную модель.

    Аргументы:
        config (dict): Словарь с конфигурацией.
    """
    logger.info("Шаг 2: Обучение модели автоэнкодера.")

    preprocess_fits_save_path = os.path.join(
        INTERIM_DATA_DIR, config.data.preprocess_fits_name
    )
    model_save_path = os.path.join(AE_OUTPUTS, config.model.save_path)
    train_loss_save_path = os.path.join(
        AE_OUTPUTS, config.interim_files.train_losses_path
    )
    val_loss_save_path = os.path.join(
        AE_OUTPUTS, config.interim_files.val_losses_path
    )
    
    if not os.path.exists(preprocess_fits_save_path):
        logger.critical(
            f"Не найден файл с предобработанными данными: {preprocess_fits_save_path}. "
            f"Сначала запустите этап подготовки данных."
        )
        exit()
    
    normalized_data_np = np.load(preprocess_fits_save_path)
    full_dataset = FITS_Dataset(normalized_data_np)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    batch_size = config.model.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=config.device== 'cuda',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=config.device== 'cuda',
    )
    
    latent_dim = config.model.latent_dim
    target_height = config.data.height
    target_width = config.data.width
    model = ConvAE(
        normalized_data_np.shape[1], latent_dim, target_height, target_width
    ).to(config.device)
    
    
    logger.info("Запущено обучение модели ConvAE...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.model.lr)
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config.model.epochs,
        device=config.device
    )
    
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Модель сохранена в: {model_save_path}")
    
    np.save(train_loss_save_path, np.array(train_losses))
    np.save(val_loss_save_path, np.array(val_losses))
    logger.info("Лоссы сохранены.")


def plot_losses_step(config):
    """
    Загружает сохранённые данные о потерях и строит график.
    
    Аргументы:
        config (AEConfig): Объект конфигурации.
    """
    logger.info("Шаг 2.1: Отрисовка графика потерь.")

    train_loss_save_path = os.path.join(
        AE_OUTPUTS, config.interim_files.train_losses_path
    )
    val_loss_save_path = os.path.join(
        AE_OUTPUTS, config.interim_files.val_losses_path
    )
    loss_plot_path = os.path.join(PLOTS_DIR, config.output.loss_plot)

    if not os.path.exists(train_loss_save_path) or not os.path.exists(val_loss_save_path):
        logger.critical(
            f"Не найдены файлы с потерями: {train_loss_save_path} или {val_loss_save_path}. "
            "Сначала запустите этап обучения модели."
        )
        exit()

    train_losses = np.load(train_loss_save_path).tolist()
    val_losses = np.load(val_loss_save_path).tolist()

    plot_losses(train_losses, val_losses, config, loss_plot_path)
    logger.info(f"График потерь сохранен в: {loss_plot_path}")


def extract_embeddings(config):
    """
    Извлекает латентные представления (эмбеддинги) из обученной модели и сохраняет их.
    
    Аргументы:
        config (AEConfig): Объект конфигурации.
    """
    logger.info("Шаг 3.1: Извлечение эмбеддингов.")

    model_save_path = os.path.join(AE_OUTPUTS, config.model.save_path)
    preprocess_fits_save_path = os.path.join(
        INTERIM_DATA_DIR, config.data.preprocess_fits_name
    )
    embeddings_save_path = os.path.join(
        PROCESSED_DATA_DIR, config.output.embeddings_csv
    )
    filepaths_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.filepaths_path
    )
    reconstructed_data_save_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.reconstructed_data_save_path
    )

    if (
        not os.path.exists(model_save_path)
        or not os.path.exists(preprocess_fits_save_path)
        or not os.path.exists(filepaths_path)
    ):
        logger.critical(
            "Необходимые файлы (модель, данные или пути) не найдены. "
            "Запустите предыдущие шаги."
        )
        exit()

    normalized_data_np = np.load(preprocess_fits_save_path)
    with open(filepaths_path, 'rb') as f:
        filepaths = pickle.load(f)

    latent_dim = config.model.latent_dim
    target_height = config.data.height
    target_width = config.data.width
    model = ConvAE(
        normalized_data_np.shape[1], latent_dim, target_height, target_width
    ).to(config.device)
    model.load_state_dict(torch.load(model_save_path, map_location=config.device))
    model.eval()
    
    full_dataset = FITS_Dataset(normalized_data_np)
    full_data_loader = DataLoader(
        full_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=config.device == 'cuda',
    )
    
    logger.info("Извлекаем и сохраняем эмбеддинги и реконструкции...")
    all_embeddings_list = []
    reconstructed_data_list = []
    with torch.no_grad():
        for data_batch in tqdm(full_data_loader, desc="Извлечение"):
            data_batch = data_batch.to(config.device)
            reconstructed_batch, latent_representation = model(data_batch)
            all_embeddings_list.append(latent_representation.cpu().numpy())
            reconstructed_data_list.append(reconstructed_batch.cpu().numpy())
            
    all_embeddings_np = np.vstack(all_embeddings_list)
    reconstructed_all_np = np.vstack(reconstructed_data_list)
    
    df_embeddings = pd.DataFrame({'filepath': filepaths})
    embedding_columns = [f'dim_{i}' for i in range(all_embeddings_np.shape[1])]
    df_embeddings = pd.concat(
        [
            df_embeddings,
            pd.DataFrame(
                all_embeddings_np, columns=embedding_columns, index=df_embeddings.index
            ),
        ],
        axis=1,
    )
    df_embeddings.to_csv(embeddings_save_path, index=False)
    logger.info(
        f"Эмбеддинги сохранены в: {embeddings_save_path} ({len(df_embeddings)} строк)"
    )

    np.save(reconstructed_data_save_path, reconstructed_all_np)
    logger.info(
        f"Реконструированные данные сохранены в: {reconstructed_data_save_path}"
    )


def visualize_reconstructions(config):
    """
    Визуализирует оригинальные и реконструированные данные.
    
    Аргументы:
        config (AEConfig): Объект конфигурации.
    """

    logger.info("Шаг 3.2: Визуализация реконструкций.")
    
    original_data_for_viz_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.original_data_for_viz_path
    )
    reconstructed_data_save_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.reconstructed_data_save_path
    )
    visualization_filenames_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.visualization_filenames_path
    )
    filepaths_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.filepaths_path
    )

    if (
        not os.path.exists(original_data_for_viz_path)
        or not os.path.exists(reconstructed_data_save_path)
        or not os.path.exists(filepaths_path)
    ):
        logger.warning("Необходимые файлы для визуализации не найдены. Визуализация пропущена.")
        return

    normalized_data_np = np.load(os.path.join(INTERIM_DATA_DIR, config.data.preprocess_fits_name))
    original_data_for_viz_np = np.load(original_data_for_viz_path)
    reconstructed_all_np = np.load(reconstructed_data_save_path)
    with open(filepaths_path, 'rb') as f:
        filepaths = pickle.load(f)

    if os.path.exists(visualization_filenames_path):
        with open(visualization_filenames_path, 'r') as f:
            viz_filepaths_to_find = [line.strip() for line in f]
        
        visual_indices = [
            filepaths.index(f) for f in viz_filepaths_to_find if f in filepaths
        ]
        
        if visual_indices:
            original_viz_data = original_data_for_viz_np[visual_indices]
            preprocessed_viz_data = normalized_data_np[visual_indices]
            reconstructed_viz_data = reconstructed_all_np[visual_indices]
            
            visualize_samples(
                original_data_np=original_viz_data,
                preprocessed_data_np=preprocessed_viz_data,
                reconstructed_data_np=reconstructed_viz_data,
                selected_filepaths=[filepaths[i] for i in visual_indices], 
                output_dir_base=PLOTS_DIR,
            )

            plot_spectral_comparison(
                preprocessed_viz_data,
                reconstructed_viz_data,
                selected_filepaths=[filepaths[i] for i in visual_indices],
                output_dir_base=PLOTS_DIR,
                target_height=config.data.height,
                target_width=config.data.width,
            )
        else:
            logger.warning(
                "Не удалось найти сохранённые файлы для визуализации в текущем "
                "наборе данных. Визуализация пропущена."
            )
    else:
        logger.warning(
            "Не удалось найти файл с именами для визуализации. Визуализация пропущена."
        )


def evaluate_reconstruction_quality(config):
    """
    Рассчитывает метрики качества (PSNR, SSIM) для реконструированных данных.
    
    Аргументы:
        config (AEConfig): Объект конфигурации.
    """
    logger.info("Шаг 3.3: Оценка качества автоэнкодера (PSNR, SSIM).")
    
    preprocess_fits_save_path = os.path.join(
        INTERIM_DATA_DIR, config.data.preprocess_fits_name
    )
    reconstructed_data_save_path = os.path.join(
        INTERIM_DATA_DIR, config.interim_files.reconstructed_data_save_path
    )

    if not os.path.exists(preprocess_fits_save_path) or not os.path.exists(reconstructed_data_save_path):
        logger.critical(
            "Не найдены файлы с предобработанными или реконструированными данными. "
            "Оценка качества невозможна."
        )
        exit()

    normalized_data_np = np.load(preprocess_fits_save_path)
    reconstructed_all_np = np.load(reconstructed_data_save_path)
    
    psnr_scores_i, ssim_scores_i, psnr_scores_v, ssim_scores_v = (
        [],
        [],
        [],
        [],
    )
    num_samples_for_metrics = min(
        normalized_data_np.shape[0], reconstructed_all_np.shape[0]
    )
    
    for i in tqdm(range(num_samples_for_metrics), desc="Расчет метрик"):
        true_i, recon_i = normalized_data_np[i, 0], reconstructed_all_np[i, 0]
        true_v, recon_v = normalized_data_np[i, 1], reconstructed_all_np[i, 1]
        data_range_i = max(np.max(true_i) - np.min(true_i), 1e-8)
        data_range_v = max(np.max(true_v) - np.min(true_v), 1e-8)
        
        win_size = min(7, true_i.shape[0], true_i.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        
        psnr_scores_i.append(
            psnr(true_i, recon_i, data_range=data_range_i)
        )
        ssim_scores_i.append(
            ssim(
                true_i,
                recon_i,
                data_range=data_range_i,
                win_size=win_size if win_size >= 3 else None,
                channel_axis=None,
            )
        )
        psnr_scores_v.append(
            psnr(true_v, recon_v, data_range=data_range_v)
        )
        ssim_scores_v.append(
            ssim(
                true_v,
                recon_v,
                data_range=data_range_v,
                win_size=win_size if win_size >= 3 else None,
                channel_axis=None,
            )
        )

    logger.info(f"Средний PSNR (I): {np.mean(psnr_scores_i):.4f}")
    logger.info(f"Средний SSIM (I): {np.mean(ssim_scores_i):.4f}")
    logger.info(f"Средний PSNR (V): {np.mean(psnr_scores_v):.4f}")
    logger.info(f"Средний SSIM (V): {np.mean(ssim_scores_v):.4f}")
