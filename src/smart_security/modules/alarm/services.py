# 模块：矩形警戒区报警服务
# 作用：按配置的矩形区域做入侵/逗留判定，与多边形 3D 主流程互补，供扩展多区域使用。

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.core.exceptions import ConfigError
from smart_security.modules.alarm.constants import AlarmType, ZoneShape
from smart_security.modules.alarm.schemas import AlarmEvent
from smart_security.modules.tracking.constants import TRACK_ROW_MIN_LEN
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ZONE_COORDINATES = [100, 100, 300, 300]
DEFAULT_LINGER_TIME = 10


def default_alarm_zones() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "name": "入口警戒区",
            "zone_type": ZoneShape.RECTANGLE.value,
            "coordinates": list(DEFAULT_ZONE_COORDINATES),
            "alarm_type": AlarmType.INTRUSION.value,
            "linger_time": DEFAULT_LINGER_TIME,
            "enabled": True,
        }
    ]


class AlarmManager:
    def __init__(self, zones_path: Path | None = None) -> None:
        settings = get_settings()
        self.zones_path = (
            Path(zones_path) if zones_path is not None else settings.alarm_zones_path
        )
        self.alarm_zones: list[dict[str, Any]] = []
        self.alarm_history: list[AlarmEvent] = []
        self.lingering_targets: dict[str, dict[str, float]] = {}
        self.load_alarm_zones()

    def load_alarm_zones(self) -> None:
        if self.zones_path.exists():
            try:
                with self.zones_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, list):
                    raise ConfigError(
                        f"alarm zones file must be a list: {self.zones_path}"
                    )
                self.alarm_zones = loaded
                logger.info("Loaded %s alarm zones", len(self.alarm_zones))
                return
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.exception("Failed to load alarm zones")
                raise ConfigError(
                    f"Invalid alarm zones file: {self.zones_path}"
                ) from exc

        logger.warning("Alarm zones file missing, writing default: %s", self.zones_path)
        self.save_alarm_zones(default_alarm_zones())
        self.alarm_zones = default_alarm_zones()

    def save_alarm_zones(self, zones: list[dict[str, Any]]) -> None:
        try:
            self.zones_path.parent.mkdir(parents=True, exist_ok=True)
            with self.zones_path.open("w", encoding="utf-8") as handle:
                json.dump(zones, handle, indent=4, ensure_ascii=False)
        except OSError as exc:
            raise ConfigError(f"Failed to save alarm zones: {self.zones_path}") from exc
        logger.info("Alarm zones saved to %s", self.zones_path)

    def add_alarm_zone(self, zone_config: dict[str, Any]) -> int:
        zone_config["id"] = len(self.alarm_zones) + 1
        self.alarm_zones.append(zone_config)
        self.save_alarm_zones(self.alarm_zones)
        return int(zone_config["id"])

    def check_intrusion(
        self, track_id: int, bbox: list[float], frame: np.ndarray
    ) -> list[AlarmEvent]:
        alarms: list[AlarmEvent] = []
        for zone in self.alarm_zones:
            if not zone.get("enabled"):
                continue
            if (
                zone.get("zone_type") != ZoneShape.RECTANGLE.value
                or zone.get("alarm_type") != AlarmType.INTRUSION.value
            ):
                continue
            if not _center_in_rect(bbox, zone["coordinates"]):
                continue
            event = AlarmEvent(
                timestamp=datetime.now().isoformat(),
                zone_id=int(zone["id"]),
                zone_name=str(zone["name"]),
                track_id=int(track_id),
                alarm_type=AlarmType.INTRUSION.value,
                bbox=list(bbox),
            )
            alarms.append(event)
            zx1, zy1, zx2, zy2 = [int(value) for value in zone["coordinates"]]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"ALARM: {zone['name']}",
                (zx1, zy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return alarms

    def check_lingering(
        self, track_id: int, bbox: list[float], frame: np.ndarray
    ) -> list[AlarmEvent]:
        alarms: list[AlarmEvent] = []
        current_time = time.time()
        for zone in self.alarm_zones:
            if (
                not zone.get("enabled")
                or zone.get("alarm_type") != AlarmType.LINGERING.value
            ):
                continue
            if zone.get("zone_type") != ZoneShape.RECTANGLE.value:
                continue
            key = f"{track_id}_{zone['id']}"
            if _center_in_rect(bbox, zone["coordinates"]):
                if key not in self.lingering_targets:
                    self.lingering_targets[key] = {
                        "enter_time": current_time,
                        "last_seen": current_time,
                    }
                    continue
                self.lingering_targets[key]["last_seen"] = current_time
                linger_time = current_time - self.lingering_targets[key]["enter_time"]
                if linger_time < float(zone["linger_time"]):
                    continue
                event = AlarmEvent(
                    timestamp=datetime.now().isoformat(),
                    zone_id=int(zone["id"]),
                    zone_name=str(zone["name"]),
                    track_id=int(track_id),
                    alarm_type=AlarmType.LINGERING.value,
                    linger_time=linger_time,
                    bbox=list(bbox),
                )
                alarms.append(event)
                zx1, zy1, zx2, zy2 = [int(value) for value in zone["coordinates"]]
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    f"LINGERING: {zone['name']}",
                    (zx1, zy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
            elif key in self.lingering_targets:
                del self.lingering_targets[key]
        return alarms

    def process_alarms(
        self, tracked_objects: list[list[float]], frame: np.ndarray
    ) -> list[AlarmEvent]:
        all_alarms: list[AlarmEvent] = []
        for obj in tracked_objects:
            if len(obj) < TRACK_ROW_MIN_LEN:
                continue
            x1, y1, x2, y2, track_id, _conf, _cls = obj[:7]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            all_alarms.extend(self.check_intrusion(int(track_id), bbox, frame))
            all_alarms.extend(self.check_lingering(int(track_id), bbox, frame))
        return all_alarms

    def save_alarm_screenshot(self, frame: np.ndarray, alarm_info: AlarmEvent) -> str:
        settings = get_settings()
        screenshots_dir = settings.screenshots_dir
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"alarm_{alarm_info.alarm_type}_{alarm_info.track_id}_{timestamp}.jpg"
        )
        filepath = screenshots_dir / filename
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"{alarm_info.alarm_type} - {alarm_info.zone_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Track ID: {alarm_info.track_id} - Time: {alarm_info.timestamp}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        written = cv2.imwrite(str(filepath), annotated)
        if not written:
            raise ConfigError(f"Failed to write alarm screenshot: {filepath}")
        return str(filepath)


def _center_in_rect(bbox: list[float], coordinates: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    zx1, zy1, zx2, zy2 = coordinates
    return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2
