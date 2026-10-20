# 模块：逗留检测状态机
# 作用：用按秒计的进出缓冲抑制边界抖动，只有稳定在区内超过阈值才触发报警；
# 时长全部取自注入的时间源，视频回放和实时流在不同帧率下行为一致。

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from smart_security.core.config import get_settings
from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.constants import ZONE_LEAVE_MIN_LINGER_SECONDS
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
        zone_in_seconds: float | None = None,
        zone_out_seconds: float | None = None,
        undetected_timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.alert_store = alert_store
        self._time = time_fn
        self.lingering_threshold = (
            lingering_threshold
            if lingering_threshold is not None
            else settings.alarm.lingering_threshold
        )
        self.zone_in_seconds = (
            zone_in_seconds if zone_in_seconds is not None else settings.zone.in_seconds
        )
        self.zone_out_seconds = (
            zone_out_seconds
            if zone_out_seconds is not None
            else settings.zone.out_seconds
        )
        self.undetected_timeout = (
            undetected_timeout
            if undetected_timeout is not None
            else settings.alarm.undetected_timeout_seconds
        )
        self.lingering_targets: dict[int, dict[str, Any]] = {}
        # 连续在区内/区外的起始时刻；短暂漏检不打断，超过 out 缓冲未见才丢弃。
        self._in_since: dict[int, float] = {}
        self._out_since: dict[int, float] = {}
        self._last_detected: dict[int, float] = {}

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
            self._last_detected[tid] = current_time

            if is_in_zone(tid):
                self._out_since.pop(tid, None)
                in_since = self._in_since.setdefault(tid, current_time)
                if tid in self.lingering_targets:
                    alert = self._update_target(tid, current_time, frame)
                    if alert is not None:
                        new_alerts.append(alert)
                elif current_time - in_since >= self.zone_in_seconds:
                    self.lingering_targets[tid] = {
                        "enter_time": in_since,
                        "last_seen": current_time,
                        "last_alert_time": in_since,
                        "alert_count": 0,
                    }
                    logger.info("Target %s entered detection zone", tid)
            else:
                self._in_since.pop(tid, None)
                self._out_since.setdefault(tid, current_time)

        self._remove_departed(current_time, detected_ids)
        return new_alerts

    def _update_target(
        self, tid: int, current_time: float, frame: np.ndarray
    ) -> AlertRecord | None:
        info = self.lingering_targets[tid]
        info["last_seen"] = current_time
        linger_time = current_time - info["enter_time"]
        if (
            linger_time < self.lingering_threshold
            or current_time - info["last_alert_time"] < self.lingering_threshold
        ):
            return None
        alert = self.alert_store.save_lingering_alert(tid, linger_time, frame)
        info["last_alert_time"] = current_time
        info["alert_count"] += 1
        logger.warning(
            "Alert #%s: target %s lingering %.1fs",
            info["alert_count"],
            tid,
            linger_time,
        )
        return alert

    def _remove_departed(self, current_time: float, detected_ids: set[int]) -> None:
        for tid in list(self.lingering_targets):
            info = self.lingering_targets[tid]
            out_since = self._out_since.get(tid)
            left_zone = (
                out_since is not None
                and current_time - out_since >= self.zone_out_seconds
            )
            vanished = (
                tid not in detected_ids
                and current_time - info["last_seen"] > self.undetected_timeout
            )
            if not (left_zone or vanished):
                continue
            linger_time = current_time - info["enter_time"]
            if linger_time > ZONE_LEAVE_MIN_LINGER_SECONDS:
                logger.info("Target %s left zone after %.1fs", tid, linger_time)
            del self.lingering_targets[tid]
            self._forget(tid)

        for tid in list(self._last_detected):
            if tid in self.lingering_targets:
                continue
            if current_time - self._last_detected[tid] > self.zone_out_seconds:
                self._forget(tid)

    def _forget(self, tid: int) -> None:
        self._in_since.pop(tid, None)
        self._out_since.pop(tid, None)
        self._last_detected.pop(tid, None)
