# 模块：报警截图读写
# 作用：从文件名解析逗留记录并提供清理，让页面层不直接碰路径规则。

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from smart_security.core.config import get_settings
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

FILENAME_PATTERNS: tuple[str, ...] = (
    r"lingering_(\d+)_(\d{8}_\d{6})\.jpg",
    r"lingering_(\d+)_(\d+)\.jpg",
    r"(\d+)_(\d{8}_\d{6})\.jpg",
)


class ScreenshotService:
    def __init__(self, screenshots_dir: Path | None = None) -> None:
        settings = get_settings()
        self.screenshots_dir = (
            Path(screenshots_dir)
            if screenshots_dir is not None
            else settings.screenshots_dir
        )
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def list_alerts(self) -> list[dict[str, Any]]:
        try:
            files = list(self.screenshots_dir.glob("*.jpg"))
        except OSError as exc:
            logger.exception("Failed to list screenshots: %s", exc)
            return []

        alerts: list[dict[str, Any]] = []
        for file in files:
            parsed = _parse_filename(file)
            alerts.append(parsed)
        logger.info("Parsed %s screenshot records", len(alerts))
        return alerts

    def delete_file(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)
        path.unlink()
        logger.info("Deleted screenshot %s", path.name)

    def clean_old(self, days: int = 7) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        count = 0
        for file in self.screenshots_dir.glob("*.jpg"):
            modified = datetime.fromtimestamp(os.path.getmtime(file))
            if modified < cutoff:
                file.unlink()
                count += 1
        logger.info("Deleted %s screenshots older than %s days", count, days)
        return count

    def clean_all(self) -> int:
        files = list(self.screenshots_dir.glob("*.jpg"))
        for file in files:
            file.unlink()
        logger.info("Deleted all %s screenshots", len(files))
        return len(files)


def _parse_filename(file: Path) -> dict[str, Any]:
    for pattern in FILENAME_PATTERNS:
        match = re.match(pattern, file.name)
        if not match:
            continue
        track_id = int(match.group(1))
        stamp = match.group(2)
        try:
            timestamp = (
                datetime.strptime(stamp, "%Y%m%d_%H%M%S")
                if "_" in stamp
                else datetime.fromtimestamp(int(stamp))
            )
        except (ValueError, OSError, OverflowError) as exc:
            logger.warning("Unreadable timestamp in %s: %s", file.name, exc)
            timestamp = datetime.fromtimestamp(os.path.getmtime(file))
        return {
            "timestamp": timestamp,
            "track_id": track_id,
            "filepath": str(file),
            "filename": file.name,
        }

    logger.warning("无法解析文件名: %s", file.name)
    return {
        "timestamp": datetime.fromtimestamp(os.path.getmtime(file)),
        "track_id": 0,
        "filepath": str(file),
        "filename": file.name,
    }
