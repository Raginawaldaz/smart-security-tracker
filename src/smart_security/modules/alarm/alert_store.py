# 模块：报警落盘
# 作用：把逗留截图和 JSON 历史写到 data/，损坏的 JSON 记录后从空列表重建，避免监控中断。

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.core.exceptions import AlertStoreError
from smart_security.modules.alarm.constants import AlarmType
from smart_security.modules.alarm.schemas import AlertRecord
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)


class AlertStore:
    def __init__(
        self,
        screenshots_dir: Path | None = None,
        alerts_file: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.screenshots_dir = (
            Path(screenshots_dir)
            if screenshots_dir is not None
            else settings.screenshots_dir
        )
        self.alerts_file = (
            Path(alerts_file) if alerts_file is not None else settings.alerts_file
        )
        self.max_history = settings.alarm.max_alerts_history
        self.save_screenshots = settings.alarm.save_screenshots
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

    def save_lingering_alert(
        self, track_id: int, linger_time: float, frame: np.ndarray
    ) -> AlertRecord:
        timestamp = datetime.now()
        filename = f"lingering_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.screenshots_dir / filename

        if self.save_screenshots:
            alert_frame = frame.copy()
            cv2.putText(
                alert_frame,
                f"LINGERING ALERT - ID: {track_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                alert_frame,
                f"Time: {linger_time:.1f}s - {timestamp.strftime('%H:%M:%S')}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            written = cv2.imwrite(str(filepath), alert_frame)
            if not written:
                raise AlertStoreError(f"Failed to write screenshot: {filepath}")

        record = AlertRecord(
            timestamp=timestamp.isoformat(),
            track_id=int(track_id),
            linger_time=round(linger_time, 1),
            screenshot=str(filepath),
            type=AlarmType.LINGERING.value,
        )
        self._append_record(record)
        return record

    def _append_record(self, record: AlertRecord) -> None:
        alerts: list[dict] = []
        if self.alerts_file.exists():
            try:
                with self.alerts_file.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list):
                    alerts = loaded
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("alerts json unreadable, rebuilding: %s", exc)
                alerts = []
        alerts.append(record.to_dict())
        alerts = alerts[-self.max_history :]
        try:
            with self.alerts_file.open("w", encoding="utf-8") as handle:
                json.dump(alerts, handle, indent=2)
        except OSError as exc:
            raise AlertStoreError(
                f"Failed to write alerts file: {self.alerts_file}"
            ) from exc
