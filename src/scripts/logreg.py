import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, roc_curve

from src import utils
from src.logger import get_logger
from src.constants import (RAW_DATA_DIR, EXTERNAL_DATA_DIR,
                           INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR)

logger = get_logger(__file__)


def prepare_logreg_data(config):
    """
    Создает и кэширует датафреймы для пайплайна логистической регрессии.

    Эта функция проверяет наличие обработанных датафреймов SHARP и RATAN.
    Если файлы не найдены, она создает их из исходных данных, добавляет целевые
    колонки и сохраняет в директорию 'interlim' для дальнейшего использования.

    Аргументы:
        config (dict): Словарь со всеми настройками пайплайна, загруженный из YAML-файла.
    """
    logger.info("Подготовка и кэширование данных...")

    sharp_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_train_save)
    sharp_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_test_save)
    ratan_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_train_save)
    ratan_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_test_save)

    if not os.path.exists(sharp_train_save_path) or not os.path.exists(sharp_test_save_path):
        logger.info("Датафреймы SHARP не найдены. Создаём и сохраняем...")
        sharp_train_df = utils.creat_sharp_df(
            config.data.sharp_years_train,
            sharps_dir=os.path.join(RAW_DATA_DIR, "sharps"),
            event_history_path=os.path.join(RAW_DATA_DIR, "events", "events_history.csv")
        )
        sharp_test_df = utils.creat_sharp_df(
            config.data.sharp_years_test,
            sharps_dir=os.path.join(RAW_DATA_DIR, "sharps"),
            event_history_path=os.path.join(RAW_DATA_DIR, "events", "events_history.csv")
        )
        
        sharp_train_df = utils.create_target_cols(sharp_train_df)
        sharp_test_df = utils.create_target_cols(sharp_test_df)
        
        sharp_train_df.to_csv(sharp_train_save_path, index=False)
        sharp_test_df.to_csv(sharp_test_save_path, index=False)
        logger.info("Датафреймы SHARP сохранены.")
    else:
        logger.info("Датафреймы SHARP найдены. Шаг пропущен.")
    
    if not os.path.exists(ratan_train_save_path) or not os.path.exists(ratan_test_save_path):
        logger.info("Датафреймы RATAN не найдены. Создаём и сохраняем...")
        ratan_train_df = utils.creat_ratan_embeddings_df(
            ratan_embeddings_path=os.path.join(PROCESSED_DATA_DIR, config.data.data_paths.ratan_embeddings),
            sync_df_path=os.path.join(EXTERNAL_DATA_DIR, "sync_train.csv")
        )
        ratan_test_df = utils.creat_ratan_embeddings_df(
            ratan_embeddings_path=os.path.join(PROCESSED_DATA_DIR, config.data.data_paths.ratan_embeddings),
            sync_df_path=os.path.join(EXTERNAL_DATA_DIR, "sync_test.csv")
        )
        ratan_train_df = utils.create_target_cols(ratan_train_df)
        ratan_test_df = utils.create_target_cols(ratan_test_df)
        
        ratan_train_df.to_csv(ratan_train_save_path, index=False)
        ratan_test_df.to_csv(ratan_test_save_path, index=False)
        logger.info("Датафреймы RATAN сохранены.")
    else:
        logger.info("Датафреймы RATAN найдены. Шаг пропущен.")


def train_and_predict_logreg(config):
    """
    Обучает модель логистической регрессии и делает предсказания.

    Функция сначала проверяет, существуют ли файлы с предсказаниями. Если да,
    шаг обучения пропускается. В противном случае, она загружает подготовленные
    данные, обучает модель с кросс-валидацией TimeSeriesSplit, делает финальные
    предсказания на тестовом наборе и сохраняет результаты в CSV-файлы.

    Аргументы:
        config (dict): Словарь со всеми настройками пайплайна.
    """
    logger.info("Обучение модели и предсказание...")

    sharp_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_train_save)
    sharp_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_test_save)
    ratan_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_train_save)
    ratan_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_test_save)
    output_csv_sharp = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_sharp)
    output_csv_ratan = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_ratan)
    
    if os.path.exists(output_csv_sharp) and os.path.exists(output_csv_ratan):
        logger.info("Файлы с предсказаниями уже существуют. Шаг пропущен.")
        return
        
    sharp_train_df = pd.read_csv(sharp_train_save_path)
    sharp_test_df = pd.read_csv(sharp_test_save_path)
    ratan_train_df = pd.read_csv(ratan_train_save_path)
    ratan_test_df = pd.read_csv(ratan_test_save_path)

    feature_cols_sharp = config.data.sharp_features
    feature_cols_ratan = config.data.ratan_features
    target_cols = config.data.target_cols
    
    base_clf = utils.get_classifier_from_config(config.model)
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_clf),
    ])
    
    tscv = TimeSeriesSplit(n_splits=config.model.cv_splits)
    
    logit_res_sharp = {}
    logit_res_ratan = {}

    for target_col in target_cols:
        logger.info(f'\n==== Обработка таргета: {target_col} ====')
        
        sharp_cv_scores = []
        for train_idx, val_idx in tscv.split(sharp_train_df):
            X_train, X_val = sharp_train_df.iloc[train_idx][feature_cols_sharp], sharp_train_df.iloc[val_idx][feature_cols_sharp]
            y_train, y_val = sharp_train_df.iloc[train_idx][target_col], sharp_train_df.iloc[val_idx][target_col]
            model = clone(base_pipe)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            sharp_cv_scores.append(score)
        logger.info(f'[SHARP] CV ROC-AUC scores: {sharp_cv_scores}')
        logger.info(f'[SHARP] Среднее: {np.mean(sharp_cv_scores):.3f}, Стандартное отклонение: {np.std(sharp_cv_scores):.3f}')
        
        final_model_sharp = clone(base_pipe)
        final_model_sharp.fit(sharp_train_df[feature_cols_sharp], sharp_train_df[target_col])
        y_sharp_proba = final_model_sharp.predict_proba(sharp_test_df[feature_cols_sharp])[:, 1]
        logit_res_sharp[target_col] = y_sharp_proba

        ratan_cv_scores = []
        for train_idx, val_idx in tscv.split(ratan_train_df):
            X_train, X_val = ratan_train_df.iloc[train_idx][feature_cols_ratan], ratan_train_df.iloc[val_idx][feature_cols_ratan]
            y_train, y_val = ratan_train_df.iloc[train_idx][target_col], ratan_train_df.iloc[val_idx][target_col]
            model = clone(base_pipe)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            ratan_cv_scores.append(score)
        logger.info(f'[RATAN] CV ROC-AUC scores: {ratan_cv_scores}')
        logger.info(f'[RATAN] Среднее: {np.mean(ratan_cv_scores):.3f}, Стандартное отклонение: {np.std(ratan_cv_scores):.3f}')

        final_model_ratan = clone(base_pipe)
        final_model_ratan.fit(ratan_train_df[feature_cols_ratan], ratan_train_df[target_col])
        y_ratan_proba = final_model_ratan.predict_proba(ratan_test_df[feature_cols_ratan])[:, 1]
        logit_res_ratan[target_col] = y_ratan_proba

    utils.save_predictions_csv(sharp_test_df.copy(), logit_res_sharp, output_csv_sharp)
    utils.save_predictions_csv(ratan_test_df.copy(), logit_res_ratan, output_csv_ratan)
    logger.info("Предсказания сохранены.")


def evaluate_and_visualize_logreg(config):
    """
    Загружает предсказания и создает графики ROC-кривых и других метрик.

    Функция проверяет, существуют ли файлы с предсказаниями. Если да, она
    загружает их, рассчитывает метрики и генерирует графики для данных SHARP
    и RATAN, сохраняя их в директорию для отчетов.

    Аргументы:
        config (dict): Словарь со всеми настройками пайплайна.
    """
    logger.info("Оценка и визуализация метрик...")

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    output_csv_sharp = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_sharp)
    output_csv_ratan = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_ratan)
    
    if not os.path.exists(output_csv_sharp) or not os.path.exists(output_csv_ratan):
        logger.error("Файлы с предсказаниями не найдены. Пожалуйста, сначала запустите этап обучения.")
        return
        
    pred_df_sharp = pd.read_csv(output_csv_sharp)
    pred_df_ratan = pd.read_csv(output_csv_ratan)

    titles_name_dict = config.output.titles
    thresholds = config.output.thresholds

    metrics_output_dir = os.path.join(PLOTS_DIR, "Forecasting_Metrics")
    os.makedirs(metrics_output_dir, exist_ok=True)
    for key in config.data.target_cols:
        utils.plot_for_flare_type(
            pred_df_sharp,
            y_pred_colname=f"logit_{key}",
            y_true_colname=f"real_{key}",
            threholds=thresholds,
            title=getattr(titles_name_dict, key),
            save_to=os.path.join(metrics_output_dir, f"Sharp_{key}.png"),
            show_fig=config.output.show_fig
        )
        utils.plot_for_flare_type(
            pred_df_ratan,
            y_pred_colname=f"logit_{key}",
            y_true_colname=f"real_{key}",
            threholds=thresholds,
            title=getattr(titles_name_dict, key),
            save_to=os.path.join(metrics_output_dir, f"Ratan_{key}.png"),
            show_fig=config.output.show_fig
        )

    roc_output_dir = os.path.join(PLOTS_DIR, "ROC_Curves")
    os.makedirs(roc_output_dir, exist_ok=True)
    for target_col in config.data.target_cols:
        
        y_sharp_true = pred_df_sharp[f"real_{target_col}"]
        y_sharp_pred = pred_df_sharp[f"logit_{target_col}"]
        y_ratan_true = pred_df_ratan[f"real_{target_col}"]
        y_ratan_pred = pred_df_ratan[f"logit_{target_col}"]
        
        sharp_auc = roc_auc_score(y_sharp_true, y_sharp_pred)
        fpr_sharp, tpr_sharp, _ = roc_curve(y_sharp_true, y_sharp_pred)
        ratan_auc = roc_auc_score(y_ratan_true, y_ratan_pred)
        fpr_ratan, tpr_ratan, _ = roc_curve(y_ratan_true, y_ratan_pred)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr_sharp, tpr_sharp, color='black', linestyle='-', lw=2, label=f'SHARP (AUC = {sharp_auc:.3f})')
        plt.plot(fpr_ratan, tpr_ratan, color='black', linestyle='--', lw=2, label=f'RATAN (AUC = {ratan_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)
        plt.title(f'ROC-кривая для: {getattr(titles_name_dict, target_col)}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_output_dir, f"ROC Curve: {target_col}"), dpi=600)
        if not config.output.show_fig:
            plt.close()
        logger.info(f"ROC-кривая для {target_col} сохранена.")
