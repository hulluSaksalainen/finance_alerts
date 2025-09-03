"""Setup and configure the logging system for the application."""
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


def setup_logging(cfg_log: Dict[str, Any]) -> logging.Logger:
    """
    Configure and return the central logger for the app.

    Features:
      - Log level configurable via config (DEBUG, INFO, WARNING, …)
      - Always logs to console (stdout)
      - Optional rotating file handler for persistent logs:
          * File size limit (maxBytes)
          * Number of backups (backupCount)
          * UTF-8 encoding for international characters

    Args:
        cfg_log: Logging configuration dictionary. Expected keys:
            - "level": str - log level (e.g. "INFO", "DEBUG")
            - "to_file": bool - whether to also log to a file
            - "file_path": str - log filename (default "alerts.log")
            - "file_max_bytes": int - max file size before rotation
            - "file_backup_count": int - number of rotated backups to keep

    Returns:
        logging.Logger: Configured logger instance named "stock-alerts".
    """
    level_name = cfg_log.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("stock-alerts")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if cfg_log.get("to_file", False):
        fh = RotatingFileHandler(
            filename=cfg_log.get("file_path", "alerts.log"),
            maxBytes=cfg_log.get("file_max_bytes", 1048576),
            backupCount=cfg_log.get("file_backup_count", 3),
            encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.debug("Logging initialized: level=%s, to_file=%s",
        level_name, cfg_log.get("to_file", False))

    return logger
