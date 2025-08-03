import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_metrics_for_all_thresholds, get_brier_score
from src.logger import get_logger

logger = get_logger(__file__)


def plot_metrics(
    thresholds: List[float],
    tss: List[float],
    pod: List[float],
    far: List[float],
    title: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """
    Строит графики метрик качества прогнозирования.

    Аргументы:
        thresholds (List[float]): Список пороговых значений.
        tss (List[float]): Список значений TSS для каждого порога.
        pod (List[float]): Список значений POD для каждого порога.
        far (List[float]): Список значений FAR для каждого порога.
        title (Optional[str]): Заголовок графика.
        suptitle (Optional[str]): Надзаголовок графика.
    """
    plt.plot(thresholds, pod, "ko-", label="Probability of discovery")
    plt.plot(thresholds, far, "k*-", label="False alarm ratio")
    plt.plot(thresholds, tss, "k--", label="True skill statistics (TSS)")

    max_pod = max(pod)
    max_far = max(far)
    max_tss = max(tss)

    plt.axhline(y=max_pod, color='k', linestyle='dotted', alpha=0.6)
    plt.axhline(y=max_far, color='k', linestyle='dotted', alpha=0.6)
    plt.axhline(y=max_tss, color='k', linestyle='dotted', alpha=0.6)

    plt.xlabel('Probability threshold')
    plt.ylabel('Score')
    plt.grid()
    plt.legend()
    plt.title(f"{title}")
    plt.suptitle(f"{suptitle}")


def plot_metrics_for_flare_type(
    data_dataframe: pd.DataFrame,
    y_pred_colname: str,
    y_true_colname: str,
    thresholds: List[float],
    title: str,
    save_to: Optional[str] = None,
):
    """
    Строит и сохраняет график метрик для определенного типа вспышек.

    Аргументы:
        data_dataframe (pd.DataFrame): DataFrame с данными.
        y_pred_colname (str): Имя колонки с предсказанными значениями.
        y_true_colname (str): Имя колонки с истинными значениями.
        thresholds (List[float]): Список пороговых значений.
        title (str): Заголовок для графика.
        save_to (Optional[str]): Путь для сохранения файла. Если None, график
                                 будет показан на экране.
    """
    tss, far, pod = get_metrics_for_all_thresholds(
        thresholds=thresholds,
        data_dataframe=data_dataframe,
        y_pred_colname=y_pred_colname,
        y_true_colname=y_true_colname,
    )
    bs = get_brier_score(
        data_dataframe=data_dataframe,
        y_pred_colname=y_pred_colname,
        y_true_colname=y_true_colname,
    )

    plot_metrics(
        thresholds=thresholds,
        tss=tss,
        far=far,
        pod=pod,
        suptitle=title,
        title=f"Brier Score: {round(bs, 4)}",
    )
    if save_to:
        plt.savefig(save_to, dpi=600)
        plt.close()
    else:
        plt.show()


def visualize_samples(
    original_data_np: np.ndarray,
    preprocessed_data_np: np.ndarray,
    reconstructed_data_np: np.ndarray,
    selected_filepaths: list,
    output_dir_base: str,
):
    """
    Визуализирует оригинальные, предобработанные и реконструированные
    изображения для выбранных файлов.
    
    Аргументы:
        original_data_np (np.ndarray): Исходные данные для выбранных файлов.
        preprocessed_data_np (np.ndarray): Предобработанные данные для выбранных файлов.
        reconstructed_data_np (np.ndarray): Реконструированные данные для выбранных файлов.
        selected_filepaths (list): Список путей к исходным файлам для визуализации.
        output_dir_base (str): Базовая директория для сохранения графиков.
    """
    logger.info("Начало визуализации выбранных примеров...")
    output_dir_visual = os.path.join(output_dir_base, "visual_reconstruction")
    os.makedirs(output_dir_visual, exist_ok=True)
    
    data_stages = {
        "Original": original_data_np,
        "Preprocessed": preprocessed_data_np,
        "Reconstructed": reconstructed_data_np,
    }

    for i in range(len(selected_filepaths)):
        filename_base = os.path.basename(selected_filepaths[i]).split(".")[0]
        logger.debug(f"Визуализация для файла: {filename_base}, порядковый индекс: {i}")

        for stage_name, data_array in data_stages.items():
            if data_array is None:
                logger.warning(
                    f"Данные для этапа '{stage_name}' отсутствуют для индекса {i}, "
                    f"пропускаем."
                )
                continue
            
            current_sample_i = data_array[i, 0]
            current_sample_v = data_array[i, 1]

            for cmap_name, suffix in [('viridis', ''), ('gray', '_gray')]:
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                fig.suptitle(
                    f'{stage_name} - {filename_base} (I and V channels)',
                    fontsize=14,
                )

                ax = axes[0]
                im = ax.imshow(current_sample_i, cmap=cmap_name, origin='lower', aspect='auto')
                ax.set_title(f'{stage_name} I')
                ax.set_xlabel('Spatial Pixels')
                ax.set_ylabel('Frequency Channels')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax = axes[1]
                im = ax.imshow(current_sample_v, cmap=cmap_name, origin='lower', aspect='auto')
                ax.set_title(f'{stage_name} V')
                ax.set_xlabel('Spatial Pixels')
                ax.set_ylabel('Frequency Channels')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                output_filename = f"{stage_name}_{filename_base}{suffix}.png"
                output_path = os.path.join(output_dir_visual, output_filename)
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                logger.debug(f"Сохранено: {output_path}")
    logger.info("Визуализация выбранных примеров завершена.")


def plot_spectral_comparison(
    preprocessed_data: np.ndarray,
    reconstructed_data: np.ndarray,
    selected_filepaths: list,   
    output_dir_base: str,
    target_height: int,
    target_width: int,
):
    """
    Строит графики сравнения спектральных профилей для выбранных файлов.

    Аргументы:
        preprocessed_data (np.ndarray): Предобработанные данные.
        reconstructed_data (np.ndarray): Реконструированные данные.
        selected_filepaths (List[str]): Список путей к исходным файлам..
        output_dir_base (str): Базовая директория для сохранения графиков.
        target_height (int): Высота изображения (количество частотных каналов).
        target_width (int): Ширина изображения (количество пространственных пикселей).
    """
    logger.info("Начало построения графиков спектрального сравнения...")
    output_dir_spectral = os.path.join(
        output_dir_base, "spectral_reconstruction_plots"
    )
    os.makedirs(output_dir_spectral, exist_ok=True)
    central_width_pixel = target_width // 2
    freq_bins = np.arange(target_height)

    for i in range(len(selected_filepaths)):
        filename_base = os.path.basename(selected_filepaths[i]).split(".")[0]
        logger.debug(
            f"Построение спектрального сравнения для файла: {filename_base}, "
            f"локальный индекс: {i}"
        )

        prep_spectrum_i = preprocessed_data[i, 0, :, central_width_pixel]
        recon_spectrum_i = reconstructed_data[i, 0, :, central_width_pixel]
        prep_spectrum_v = preprocessed_data[i, 1, :, central_width_pixel]
        recon_spectrum_v = reconstructed_data[i, 1, :, central_width_pixel]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f'Spectral Profile Comparison - {filename_base}\n(Central Spatial Pixel)',
            fontsize=14,
        )

        axes[0].plot(
            freq_bins,
            prep_spectrum_i,
            color='k',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes[0].plot(
            freq_bins, recon_spectrum_i, color='r', linestyle='--', label='Reconstructed'
        )
        axes[0].set_title(f'Channel I')
        axes[0].set_xlabel('Frequency channel')
        axes[0].set_ylabel('Normalized Value')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            freq_bins,
            prep_spectrum_v,
            color='k',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes[1].plot(
            freq_bins, recon_spectrum_v, color='r', linestyle='--', label='Reconstructed'
        )
        axes[1].set_title(f'Channel V')
        axes[1].set_xlabel('Frequency channel')
        axes[1].set_ylabel('Normalized Value')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        output_filename = f"Spectrum_Comparison_{filename_base}.png"
        output_path = os.path.join(output_dir_spectral, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        logger.debug(f"Сохранено: {output_path}")

        # Grayscale версия
        fig_gray, axes_gray = plt.subplots(1, 2, figsize=(16, 6))
        fig_gray.suptitle(
            f'Spectral Profile Comparison (Grayscale) - {filename_base}\n'
            f'(Central Spatial Pixel)',
            fontsize=14,
        )

        axes_gray[0].plot(
            freq_bins,
            prep_spectrum_i,
            color='black',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes_gray[0].plot(
            freq_bins,
            recon_spectrum_i,
            color='dimgray',
            linestyle='--',
            label='Reconstructed',
        )
        axes_gray[0].set_title(f'Channel I')
        axes_gray[0].set_xlabel('Frequency Bin (Height Dimension Index)')
        axes_gray[0].set_ylabel('Normalized Value')
        axes_gray[0].legend()
        axes_gray[0].grid(True)

        axes_gray[1].plot(
            freq_bins,
            prep_spectrum_v,
            color='black',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes_gray[1].plot(
            freq_bins,
            recon_spectrum_v,
            color='dimgray',
            linestyle='--',
            label='Reconstructed',
        )
        axes_gray[1].set_title(f'Channel V')
        axes_gray[1].set_xlabel('Frequency Bin (Height Dimension Index)')
        axes_gray[1].set_ylabel('Normalized Value')
        axes_gray[1].legend()
        axes_gray[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        output_filename_gray = f"Spectrum_Comparison_{filename_base}_gray.png"
        output_path_gray = os.path.join(output_dir_spectral, output_filename_gray)
        plt.savefig(output_path_gray, bbox_inches='tight', dpi=100)
        plt.close(fig_gray)
        logger.debug(f"Сохранено: {output_path_gray}")
    logger.info("Построение графиков спектрального сравнения завершено.")
