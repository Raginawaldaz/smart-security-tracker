# 模块：逗留检测状态机
# 作用：用进出帧缓冲抑制边界抖动，只有稳定在区内超过阈值才触发报警。

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from smart_security.core.config import get_settings
from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.constants import (
    ZONE_LEAVE_MIN_LINGER_SECONDS,
    ZONE_UNDETECTED_TIMEOUT_SECONDS,
)
from smart_security.modules.alarm.schemas import AlertRecord
from smart_security.modules.tracking.constants import PERSON_CLASS_ID, TRACK_ROW_MIN_LEN
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

InZoneFn = Callable[[int], bool]
TimeFn = Callable[[], float]


class LingeringDetector:
    def __init__(
        self,
        alert_store: AlertStore,
        time_fn: TimeFn,
        lingering_threshold: float | None = None,
        zone_in_frames: int | None = None,
        zone_out_frames: int | None = None,
    ) -> None:
        settings = get_settings()
        self.alert_store = alert_store
        self._time = time_fn
        self.lingering_threshold = (
            lingering_threshold
            if lingering_threshold is not None
            else settings.alarm.lingering_threshold
        )
        self.zone_in_frames = (
            zone_in_frames if zone_in_frames is not None else settings.zone.in_frames
        )
        self.zone_out_frames = (
            zone_out_frames if zone_out_frames is not None else settings.zone.out_frames
        )
        self.lingering_targets: dict[int, dict[str, Any]] = {}
        self._zone_in_counter: dict[int, int] = {}
        self._zone_out_counter: dict[int, int] = {}

    def check(
        self,
        tracked_objects: Iterable[Sequence[float]],
        frame: np.ndarray,
        is_in_zone: InZoneFn,
    ) -> list[AlertRecord]:
        current_time = self._time()
        new_alerts: list[AlertRecord] = []
        detected_ids: set[int] = set()

        for obj in tracked_objects:
            if len(obj) < TRACK_ROW_MIN_LEN:
                continue
            _x1, _y1, _x2, _y2, track_id, _conf, cls = obj[:7]
            if int(cls) != PERSON_CLASS_ID:
                continue

            tid = int(track_id)
            detected_ids.add(tid)
            in_zone = is_in_zone(tid)

            if in_zone:
                self._zone_in_counter[tid] = self._zone_in_counter.get(tid, 0) + 1
                self._zone_out_counter[tid] = 0

                if self._zone_in_counter[tid] >= self.zone_in_frames:
                    if tid not in self.lingering_targets:
                        self.lingering_targets[tid] = {
                            "enter_time": current_time,
                            "last_seen": current_time,
                            "last_alert_time": current_time,
                            "alert_count": 0,
                        }
                        logger.info("Target %s entered detection zone", tid)
                    else:
                        info = self.lingering_targets[tid]
                        info["last_seen"] = current_time
                        linger_time = current_time - info["enter_time"]
                        if (
                            linger_time >= self.lingering_threshold
                            and current_time - info["last_alert_time"]
                            >= self.lingering_threshold
                        ):
                            alert = self.alert_store.save_lingering_alert(
                                tid, linger_time, frame
                            )
                            new_alerts.append(alert)
                            info["last_alert_time"] = current_time
                            info["alert_count"] += 1
                            logger.warning(
                                "Alert #%s: target %s lingering %.1fs",
                                info["alert_count"],
                                tid,
                                linger_time,
                            )
            else:
                self._zone_out_counter[tid] = self._zone_out_counter.get(tid, 0) + 1
                self._zone_in_counter[tid] = 0

        to_remove: list[int] = []
        for tid in list(self.lingering_targets):
            out_frames = self._zone_out_counter.get(tid, 0)
            not_detected = tid not in detected_ids
            last_seen = self.lingering_targets[tid]["last_seen"]
            if out_frames >= self.zone_out_frames or (
                not_detected
                and current_time - last_seen > ZONE_UNDETECTED_TIMEOUT_SECONDS
            ):
                linger_time = current_time - self.lingering_targets[tid]["enter_time"]
                if linger_time > ZONE_LEAVE_MIN_LINGER_SECONDS:
                    logger.info("Target %s left zone after %.1fs", tid, linger_time)
                to_remove.append(tid)
        for tid in to_remove:
            del self.lingering_targets[tid]
            self._zone_in_counter.pop(tid, None)
            self._zone_out_counter.pop(tid, None)

        for tid in list(self._zone_in_counter):
            if tid not in detected_ids:
                self._zone_in_counter.pop(tid, None)
                self._zone_out_counter.pop(tid, None)

        return new_alerts
