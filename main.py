import os
import matplotlib.pyplot as plt
from src.logger import get_logger
from src.constants import create_dirs, CONFIG_DIR
from src.scripts import logreg as logreg_pipeline_steps
from src.scripts import embeddings as ae_pipeline_steps
from src.utils import load_config, load_logreg_config, load_ae_config

plt.switch_backend('Agg')

logger = get_logger(__file__)


def logreg_pipe():
    """
    Основной исполнительный блок для пайплайна логистической регрессии.

    Эта функция управляет всем пайплайном, загружая конфигурацию
    и последовательно вызывая модульные шаги: подготовку данных,
    обучение/предсказание и оценку/визуализацию.
    """
    logger.info("Запуск пайплайна логистической регрессии.")

    # Загрузка конфигурации
    config = load_logreg_config(os.path.join(CONFIG_DIR, "logreg_config.yaml"))

    # 1. Подготовка данных
    logreg_pipeline_steps.prepare_logreg_data(config)

    # 2. Обучение и предсказание
    logreg_pipeline_steps.train_and_predict_logreg(config)

    # 3. Оценка и визуализация
    logreg_pipeline_steps.evaluate_and_visualize_logreg(config)

    logger.info("Пайплайн логистической регрессии успешно завершен.")


def ae_pipe():
    """
    Главный исполнительный блок для пайплайна автоэнкодера.

    Эта функция управляет всем пайплайном, загружая конфигурацию
    и последовательно вызывая модульные шаги: подготовку данных,
    обучение/загрузку модели и извлечение эмбеддингов/оценку качества.
    """
    logger.info("Запуск пайплайна автоэнкодера.")

    config_path = os.path.join(CONFIG_DIR, "ae_config.yaml")
    config = load_ae_config(config_path)

    # 1. Подготовка данных: скачивание, предобработка и сохранение
    ae_pipeline_steps.download_and_prepare_data(config)

    # 2.1 Обучение или загрузка модели
    if config.model.train_model:
        ae_pipeline_steps.train_ae_model(config)

    # 2.2 Визуализация функции потерь
    if config.output.plot_losses:
        ae_pipeline_steps.plot_losses_step(config)

    # 3.1 Извлечение эмбеддингов
    if config.output.extract_embeddings:
        ae_pipeline_steps.extract_embeddings(config)

    # 3.2 Визуализация реконструкций
    if config.output.visualize_reconstructions:
        ae_pipeline_steps.visualize_reconstructions(config)

    # 3.3 Оценка качества реконструкций
    if config.output.evaluate_reconstructions:
        ae_pipeline_steps.evaluate_reconstruction_quality(config)

    logger.info("Пайплайн автоэнкодера успешно завершен.")


def main(config):
    """
    Основная функция для запуска пайплайнов.
    """
    create_dirs()

    if config["run_ae_pipeline"]:
        ae_pipe()
    else:
        logger.info("AE пайплайн пропущен.")

    if config["run_logreg_pipeline"]:
        logreg_pipe()
    else:
        logger.info("LogReg пайплайн пропущен.")


if __name__ == "__main__":
    config = load_config(os.path.join(CONFIG_DIR, "main_config.yaml"))
    main(config)