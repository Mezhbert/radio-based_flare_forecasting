import os
import ast
import yaml
from tqdm import tqdm
from typing import List, Tuple, Any
from pydantic import BaseModel, ValidationError

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, brier_score_loss

from src.logger import get_logger
from src.config_models import AEConfig, LogRegConfig

logger = get_logger(__file__)


def load_config(path: str) -> Any:
    """
    Загружает конфигурацию из YAML-файла.

    Эта функция загружает содержимое YAML-файла и возвращает его
    в виде словаря или другого объекта. Валидация данных не выполняется.

    Args:
        path (str): Путь к YAML-файлу.

    Returns:
        Any: Содержимое файла, обычно в виде словаря.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_ae_config(path: str) -> AEConfig:
    """
    Загружает и валидирует файл конфигурации автоэнкодера.

    Аргументы:
        path (str): Путь к YAML-файлу.

    Возвращает:
        AEConfig: Валидированный объект конфигурации.
    
    Исключения:
        ValidationError: Если данные в файле не соответствуют модели.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return AEConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Ошибка валидации конфигурации ae_config.yaml:\n{e}")
        raise


def load_logreg_config(path: str) -> LogRegConfig:
    """
    Загружает и валидирует файл конфигурации логистической регрессии.

    Аргументы:
        path (str): Путь к YAML-файлу.

    Возвращает:
        LogRegConfig: Валидированный объект конфигурации.
    
    Исключения:
        ValidationError: Если данные в файле не соответствуют модели.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return LogRegConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Ошибка валидации конфигурации logreg_config.yaml:\n{e}")
        raise




def plot_losses(
    train_losses: List[float], val_losses: List[float], config: BaseModel, path: str
):
    """
    Строит и сохраняет график потерь обучения и валидации.

    Аргументы:
        train_losses (List[float]): Список значений потерь на обучающей выборке.
        val_losses (List[float]): Список значений потерь на валидационной выборке.
        config (BaseModel): Объект конфигурации для настроек цветов и стиля.
        path (str): Путь для сохранения графика.
    """
    color_train = config.output.train_color
    color_val = config.output.val_color
    grayscale = config.output.grayscale

    if grayscale:
        color_train = "black"
        color_val = "dimgray"

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Потери на обучении', linestyle='-', color=color_train)
    plt.plot(
        val_losses, label='Потери на валидации', linestyle='--', color=color_val
    )
    plt.title('Потери на этапах обучения и валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение потерь (MSE)')

    if len(train_losses) <= 20:
        plt.xticks(range(1, len(train_losses) + 1))
    else:
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    logger.info(f"График потерь сохранен в: {path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: str,
) -> Tuple[List[float], List[float]]:
    """
    Выполняет обучение модели автоэнкодера.

    Аргументы:
        model (nn.Module): Модель автоэнкодера.
        train_loader (DataLoader): Загрузчик данных для обучающей выборки.
        val_loader (DataLoader): Загрузчик данных для валидационной выборки.
        criterion (nn.Module): Функция потерь.
        optimizer (optim.Optimizer): Оптимизатор.
        num_epochs (int): Количество эпох для обучения.

    Возвращает:
        tuple: Кортеж со списками значений потерь на обучении и валидации.
    """
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader,
                         desc=f"Эпоха {epoch + 1}/{num_epochs} [Обучение]"):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader,
                             desc=f"Эпоха {epoch + 1}/{num_epochs} [Валидация]"):
                data = data.to(device)
                reconstructed, _ = model(data)
                val_loss = criterion(reconstructed, data)
                val_running_loss += val_loss.item()
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        logger.info(
            f"Эпоха {epoch + 1}: Потери на обучении = {train_loss:.6f}, "
            f"Потери на валидации = {val_loss:.6f}"
        )

    return train_losses, val_losses




def creat_sharp_df(years: list[int], sharps_dir: str, event_history_path: str) -> pd.DataFrame:
    """works only with Sharps csv files"""

    dfs = []
    for y in years:
        sharp_csv = os.path.join(sharps_dir, f'{y}Sharp.csv')
        dfs.append(pd.read_csv(sharp_csv))
    df = pd.concat(dfs)[['ratan_filename', 'NOAA_num','harp','T_REC', 'CRVAL1', 'CRLN_OBS', 'USFLUX', 'MEANGBT', 'MEANJZH',
        'MEANPOT', 'SHRGT45', 'TOTUSJH', 'MEANGBH', 'MEANALP', 'MEANGAM',
        'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'SAVNCPP', 'TOTPOT', 'MEANSHR',
        'AREA_ACR', 'R_VALUE', 'ABSNJZH']]
    df_events = pd.read_csv(event_history_path)
    data_df = df[~df['T_REC'].isna()].join(df_events.set_index('key'), on='ratan_filename')
    data_df["T_REC"] = data_df["T_REC"].str.replace("_TAI", "", regex=False)
    data_df["T_REC"] = pd.to_datetime(data_df ["T_REC"], format="%Y.%m.%d_%H:%M:%S")
    data_df = data_df.dropna().sort_values("T_REC").reset_index(drop=True)
    
    logger.info(f"SHARP df for years {years} is {data_df.shape}")

    return data_df


def creat_ratan_embeddings_df(ratan_embeddings_path: str, sync_df_path: str):
    sync_df = pd.read_csv(sync_df_path)
    all_embeddings_df = pd.read_csv(ratan_embeddings_path)

    data_df = all_embeddings_df.copy()

    data_df['filepath'] = data_df['filepath'].apply(lambda x: x.split('/')[-1])

    logger.info(f"embeddings full shape: {all_embeddings_df.shape}")

    data_df = data_df[data_df['filepath'].isin(sync_df['ratan_filename'])]

    data_df = pd.merge(data_df, sync_df[['ratan_filename', 'day', 'day after']],
                    left_on='filepath', right_on='ratan_filename',
                    how='left')
    data_df = data_df.drop(columns=['filepath'])

    logger.info(f"ratan embeddings data shape for : {data_df.shape}")

    return data_df


def parse_spectrum_filename(filename):
    """
    Парсит имя файла спектра для извлечения даты, времени и номера активной области.
    Пример имени файла: 20110103_075046_AR1140_20.0.fits
    """
    import re
    from datetime import datetime

    match = re.match(r"(\d{8})_(\d{6})_AR(\d+)_.*\.fits", filename)
    if not match:
        raise ValueError(f"Wrong file name format: {filename}")
    
    date_str, time_str, ar_str = match.groups()
    
    spectrum_datetime_str = date_str + time_str
    spectrum_dt = datetime.strptime(spectrum_datetime_str, "%Y%m%d%H%M%S")
    ar_number = int('1'+ar_str)
    
    return spectrum_dt, ar_number

def num_flare_class(flares: List[str], f_cl ='M'):
    """
    If there is C and M or X
    Transform list of flares ['C1.1', 'M1.0', 'X2.1'] → [1, 1]
    """
    if isinstance(flares, str):
        flares = ast.literal_eval(flares)

    return 1*(sum(f_cl in str(event) for event in flares)>0)


def create_target_cols(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df['is_C_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='C'))
    data_df['is_M_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='M'))
    data_df['is_X_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='X'))
    data_df['is_C_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='C'))
    data_df['is_M_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='M'))
    data_df['is_X_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='X'))
    data_df['Mplus_24'] =  1.*(data_df['is_M_24']+data_df['is_X_24']>0)
    data_df['Mplus_48'] =  1.*(data_df['is_M_48']+data_df['is_X_48']>0)

    return data_df

def compute_stats(TP, TN, FP, FN):
    TSS = (TP/(TP+FN))-(FP/(FP+TN))
    POD = TP/(TP+FN)
    if (FP+TP)==0:
        FAR = 0
    else:
        FAR = FP/(FP+TP)

    return TSS, POD, FAR

def get_stat_by_df(data_dataframe: pd.DataFrame, 
                   y_pred_colname: str, 
                   y_true_colname: str, 
                   thres=0.4):
    
    tn, fp, fn, tp = confusion_matrix(data_dataframe[y_true_colname], 
                                      1.*(data_dataframe[y_pred_colname]>thres)).ravel()
    TSS, POD, FAR = compute_stats(tp, tn, fp, fn)

    return {'TSS': TSS, 'POD': POD, 'FAR': FAR}


def get_brier_score(data_dataframe: pd.DataFrame, 
                    y_pred_colname: str, 
                    y_true_colname: str):

    return brier_score_loss(data_dataframe[y_true_colname], data_dataframe[y_pred_colname])


def get_metrics_for_all_thresholds(thresholds: List[int], 
                                   data_dataframe: pd.DataFrame, 
                                   y_pred_colname: str, 
                                   y_true_colname: str,
                                   return_lists=True):
    """
    If return_lists (by default: True) -> return lists for scores 
    in the following order: TSS, FAR, POD

    else -> return a dict of scores where: 
        key: thresh 
        value: a dict of scores returned by get_stat_by_df  
    """
    res = {}
    for thresh in thresholds:
        res[str(thresh)] = get_stat_by_df(data_dataframe, 
                                          y_pred_colname, 
                                          y_true_colname, 
                                          thres=thresh)
    if return_lists:
        TSS, FAR, POD = [], [], []

        for thresh in thresholds:
            TSS.append(res[str(thresh)]['TSS'])
            POD.append(res[str(thresh)]['POD'])
            FAR.append(res[str(thresh)]['FAR'])
        
        return TSS, FAR, POD
    else:
        return res

def draw_stat_plot(thrs, TSS, POD, FAR, title: str =None, suptitle: str =None):
    plt.plot(thrs, POD, "ko-", label = f"Probability of discovery" );
    plt.plot(thrs, FAR, "k*-", label = f"False alarm ratio");
    plt.plot(thrs, TSS, "k--", label = "True skill statistics (TSS)");
    max_pod = max(POD)
    max_far = max(FAR)
    max_tss = max(TSS)

    plt.axhline(y=max_pod, color='k', linestyle='dotted', alpha=0.6);
    plt.axhline(y=max_far, color='k', linestyle='dotted', alpha=0.6);
    plt.axhline(y=max_tss, color='k', linestyle='dotted', alpha=0.6);

    plt.xlabel('Probability threshold');
    plt.ylabel('Score');

    plt.grid();
    plt.legend();
    plt.title(f"{title}");
    plt.suptitle(f"{suptitle}");

def plot_for_flare_type(data_dataframe: pd.DataFrame,
                        y_pred_colname: str, 
                        y_true_colname: str, 
                        threholds: List[int],
                        title: str,
                        save_to: str = None,
                        show_fig: bool = False):
    TSS, FAR, POD = get_metrics_for_all_thresholds(thresholds=threholds,
                                                    data_dataframe=data_dataframe, 
                                                    y_pred_colname=y_pred_colname, 
                                                    y_true_colname=y_true_colname)
    BS = get_brier_score(data_dataframe=data_dataframe, 
                            y_pred_colname=y_pred_colname, 
                            y_true_colname=y_true_colname)

    draw_stat_plot(thrs=threholds, 
                    TSS=TSS, 
                    FAR=FAR, 
                    POD=POD, 
                    suptitle=title, 
                    title=f"Brier Score: {round(BS, 4)}")
    if show_fig:
        plt.show()
    
    plt.close()

def save_predictions_csv(
    data_dataframe: pd.DataFrame, predictions_dict: dict, output_path: str
):
    """
    Сохраняет предсказания в CSV-файл.

    Функция создаёт DataFrame с истинными значениями и предсказанными
    логитами, а затем сохраняет его в указанный путь.

    Аргументы:
        data_dataframe (pd.DataFrame): Исходный DataFrame с данными.
        predictions_dict (dict): Словарь, где ключ — целевая переменная,
                                 значение — предсказанные логиты.
        output_path (str): Путь для сохранения CSV-файла.
    """
    results_df = pd.DataFrame()

    for target_col, preds_logit in predictions_dict.items():
        results_df[f'logit_{target_col}'] = preds_logit
        results_df[f'real_{target_col}'] = data_dataframe[target_col]

    if 'filename' in data_dataframe.columns:
        results_df['filename'] = data_dataframe['filename']
    elif 'ratan_filename' in data_dataframe.columns:
        results_df['ratan_filename'] = data_dataframe['ratan_filename']
        
    results_df.to_csv(output_path, index=False)
    logger.info(f"Предсказания сохранены в {output_path}")


def gather_fits_files(base_data_dir: str, target_years: list = None, default_year_subdir: str = "2025") -> list:
    """
    Gathers all .fits file paths from specified year subdirectories.
    """
    fits_files = []
    
    if not target_years:
        years_to_scan = [default_year_subdir]
    else:
        years_to_scan = target_years

    for year in years_to_scan:
        year_dir = os.path.join(base_data_dir, year)
        if os.path.isdir(year_dir):
            logger.info(f"Scanning for FITS files in: {year_dir}")
            for filename in sorted(os.listdir(year_dir)):
                if filename.lower().endswith((".fits", ".fit")):
                    fits_files.append(os.path.join(year_dir, filename))
        else:
            logger.warning(f"Directory not found - {year_dir}")
            
    if not fits_files:
        logger.warning(f"No FITS files found in specified directories: {(', '.join(years_to_scan)) if years_to_scan else 'default'}")
    else:
        logger.info(f"Found {len(fits_files)} FITS files.")
    return fits_files


def get_classifier_from_config(model_config):
    """
    Создает объект классификатора на основе конфигурации.
    """
    model_name = model_config.name
    model_params = model_config.params
    
    if hasattr(model_params, 'to_dict'):
        model_params_dict = model_params.to_dict()
    else:
        model_params_dict = {
            'class_weight': model_params.class_weight,
            'random_state': model_params.random_state,
        }

    if model_name == "LogisticRegression":
        return LogisticRegression(**model_params_dict)
    else:
        raise ValueError(f"Неподдерживаемый классификатор: {model_name}")