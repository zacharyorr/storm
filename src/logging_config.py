# src/logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "pipeline_logger",
    log_file: str = "./logs/pipeline.log",
    level: int = logging.DEBUG,
    max_bytes: int = 5_000_000,
    backup_count: int = 3
):
    """
    Configures a logger with both console and rotating file handlers.
    
    :param name: The name of the logger
    :param log_file: Path to the log file
    :param level: Log level (DEBUG, INFO, WARNING, etc.)
    :param max_bytes: Max size of log file in bytes before rotation
    :param backup_count: Number of old log files to keep
    :return: Configured logger
    """

    # Make sure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(ch_format)
        logger.addHandler(ch)
        
        # Rotating file handler
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fh_format)
        logger.addHandler(fh)
    
    return logger
