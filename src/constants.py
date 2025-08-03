import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

BASE_DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DATA_DIR, "interim")
EXTERNAL_DATA_DIR = os.path.join(BASE_DATA_DIR, "external")
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed")

BASE_LOG_DIR = os.path.join(ROOT_DIR, "log")

AE_OUTPUTS = os.path.join(ROOT_DIR, "models", "conv_ae")
LOGREG_OUTPUTS = os.path.join(ROOT_DIR, "models", "logreg")

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "figures")

def create_dirs():
    """Создает все необходимые директории, если они не существуют."""
    dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        EXTERNAL_DATA_DIR,
        PROCESSED_DATA_DIR,
        BASE_LOG_DIR,
        AE_OUTPUTS,
        LOGREG_OUTPUTS,
        REPORTS_DIR,
        PLOTS_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
