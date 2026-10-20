# 模块：警戒区成员判定
# 作用：用足点 + 下半身覆盖率经 EMA 与 Schmitt 滞回判断 track 是否在区内，
# 避免身体跨边界时整框在绿/非绿之间来回跳。

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from smart_security.core.config import get_settings
from smart_security.modules.tracking.constants import PERSON_CLASS_ID, TRACK_ROW_MIN_LEN
from smart_security.modules.zone.zone_config import ZoneConfig

FOOT_SCORE_WEIGHT = 0.45
COVERAGE_SCORE_WEIGHT = 0.55
COVERAGE_SOFTMAX_SCALE = 0.6


class ZoneMembership:
    def __init__(self, zone: ZoneConfig) -> None:
        settings = get_settings()
        self.zone = zone
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        self.lower_body_ratio = settings.zone.lower_body_ratio
        self.score_ema_alpha = settings.zone.score_ema_alpha
        self.schmitt_high = settings.zone.schmitt_high
        self.schmitt_low = settings.zone.schmitt_low
        self.membership_ttl_frames = settings.zone.membership_ttl_frames
        self._score_ema: dict[int, float] = {}
        self._hyst_in: dict[int, bool] = {}
        self._last_seen_frame: dict[int, int] = {}

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height

    def geometry_score(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, bool]:
        width = self.frame_width or 1
        height = self.frame_height or 1
        foot_x = (float(x1) + float(x2)) * 0.5
        foot_y = float(y2)
        foot_in = self.zone.point_in_zone(foot_x, foot_y, width, height)
        cover = self.zone.coverage_ratio(
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            width,
            height,
            lower_band=self.lower_body_ratio,
        )
        score = FOOT_SCORE_WEIGHT * (
            1.0 if foot_in else 0.0
        ) + COVERAGE_SCORE_WEIGHT * float(
            np.clip(cover / COVERAGE_SOFTMAX_SCALE, 0.0, 1.0)
        )
        return float(np.clip(score, 0.0, 1.0)), bool(foot_in)

    def update(
        self, tracked_objects: Iterable[Sequence[float]], frame_count: int
    ) -> None:
        seen: set[int] = set()
        for obj in tracked_objects:
            if len(obj) < TRACK_ROW_MIN_LEN:
                continue
            x1, y1, x2, y2, track_id, _conf, cls = obj[:7]
            if int(cls) != PERSON_CLASS_ID:
                continue
            tid = int(track_id)
            seen.add(tid)
            geometry, _foot = self.geometry_score(x1, y1, x2, y2)
            previous = self._score_ema.get(tid, geometry)
            ema = (
                self.score_ema_alpha * geometry
                + (1.0 - self.score_ema_alpha) * previous
            )
            self._score_ema[tid] = ema
            self._last_seen_frame[tid] = frame_count

            previous_hyst = self._hyst_in.get(tid, False)
            if ema >= self.schmitt_high:
                self._hyst_in[tid] = True
            elif ema <= self.schmitt_low:
                self._hyst_in[tid] = False
            else:
                self._hyst_in[tid] = previous_hyst

        for tid in list(self._score_ema):
            if tid in seen:
                continue
            last_frame = self._last_seen_frame.get(tid, frame_count)
            if frame_count - last_frame > self.membership_ttl_frames:
                self._score_ema.pop(tid, None)
                self._hyst_in.pop(tid, None)
                self._last_seen_frame.pop(tid, None)

    def is_in_zone(self, track_id: int) -> bool:
        return bool(self._hyst_in.get(int(track_id), False))
