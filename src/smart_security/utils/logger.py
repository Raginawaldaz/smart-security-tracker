# 模块：日志初始化
# 作用：统一 logging 格式与文件轮转，禁止业务代码使用 print。

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from smart_security.core.config import PROJECT_ROOT, get_settings

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(
    level: str | None = None,
    log_dir: Path | None = None,
    force: bool = False,
) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    log_level_name = (level or settings.system.log_level).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    directory = Path(log_dir) if log_dir is not None else settings.logs_dir
    directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        directory / "smart_security.log",
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("smart_security").setLevel(log_level)
    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "Logging initialized at %s (root=%s)", log_level_name, PROJECT_ROOT
    )
