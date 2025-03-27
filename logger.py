import logging
import os
from threading import Lock

_log_lock = Lock()

def setup_logger(log_path='logs/morse_system.log'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger('MorseSystem')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    with _log_lock:
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    return logger

logger = setup_logger()

