import logging
import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_logger(source_file: str, output_dir: str = os.path.join(ROOT_DIR, "log")) -> logging.Logger:
    """
    Создает и возвращает логгер, который выводит сообщения в консоль
    и в файл.

    Args:
        source_file (str): Путь к исходному файлу, для которого
                           создается логгер (используется для имени).
        output_dir (str): Директория, куда будет сохранен лог-файл.

    Returns:
        logging.Logger: Настроенный логгер.
    """
    filename = os.path.basename(source_file)
    name, _ = os.path.splitext(filename)

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
