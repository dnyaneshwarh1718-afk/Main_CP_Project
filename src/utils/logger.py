import logging
from datetime import datetime
from src.config.settings import LOG_DIR

def setup_logger(name:str):
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler()

        fmt = logging.Formatter("[%(asctime)s %(levelname)s -%(message)s]")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger

