# 模块：人员轨迹
# 作用：用足底略上方锚点记录平滑尾迹，短时漏检不立刻清掉，减少轨迹断裂。

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.modules.tracking.constants import (
    LINGERING_ORANGE,
    LINGERING_RED,
    PERSON_CLASS_ID,
    TRACK_ROW_MIN_LEN,
    TRAJECTORY_OUTLINE_COLOR,
)

Point = tuple[float, float]


class TrajectoryStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.person_tracks: dict[int, list[Point]] = {}
        self.track_miss_counts: dict[int, int] = {}
        self.ema_alpha = settings.trajectory.ema_alpha
        self.min_move_px = settings.trajectory.min_move_px
        self.keep_missing = settings.trajectory.keep_missing
        self.max_len = settings.trajectory.max_len
        self.anchor_bottom_lift_ratio = settings.trajectory.anchor_bottom_lift_ratio
        self.anchor_bottom_lift_min_px = settings.trajectory.anchor_bottom_lift_min_px
        self.anchor_lift_max_ratio = settings.trajectory.anchor_lift_max_ratio

    def trajectory_anchor(self, x1: float, y1: float, x2: float, y2: float) -> Point:
        """人体底部略上方的画面锚点：比纯 y2 更易读、不易贴边。"""
        box_height = max(float(y2) - float(y1), 1.0)
        lift = max(
            self.anchor_bottom_lift_min_px,
            self.anchor_bottom_lift_ratio * box_height,
        )
        lift = min(lift, self.anchor_lift_max_ratio * box_height)
        center_x = (float(x1) + float(x2)) * 0.5
        center_y = float(y2) - lift
        return center_x, center_y

    def update(self, tracked_objects: Iterable[Sequence[float]]) -> None:
        seen_ids: set[int] = set()
        for obj in tracked_objects:
            if len(obj) < TRACK_ROW_MIN_LEN:
                continue
            x1, y1, x2, y2, track_id, _conf, cls = obj[:7]
            if int(cls) != PERSON_CLASS_ID:
                continue

            center_x, center_y = self.trajectory_anchor(x1, y1, x2, y2)
            tid = int(track_id)
            seen_ids.add(tid)
            self.track_miss_counts[tid] = 0

            if tid not in self.person_tracks:
                self.person_tracks[tid] = [(center_x, center_y)]
                continue

            last_x, last_y = self.person_tracks[tid][-1]
            smooth_x = self.ema_alpha * center_x + (1.0 - self.ema_alpha) * last_x
            smooth_y = self.ema_alpha * center_y + (1.0 - self.ema_alpha) * last_y

            if np.hypot(smooth_x - last_x, smooth_y - last_y) >= self.min_move_px:
                self.person_tracks[tid].append((smooth_x, smooth_y))
                if len(self.person_tracks[tid]) > self.max_len:
                    self.person_tracks[tid].pop(0)

        for tid in list(self.person_tracks):
            if tid in seen_ids:
                continue
            self.track_miss_counts[tid] = self.track_miss_counts.get(tid, 0) + 1
            if self.track_miss_counts[tid] > self.keep_missing:
                del self.person_tracks[tid]
                self.track_miss_counts.pop(tid, None)

    def last_point(self, track_id: int) -> Point | None:
        points = self.person_tracks.get(int(track_id))
        if not points:
            return None
        return points[-1]

    def draw(self, frame: np.ndarray, is_in_zone) -> None:
        for track_id, positions in self.person_tracks.items():
            if not positions:
                continue
            in_zone = is_in_zone(int(track_id))
            base_color = LINGERING_RED if in_zone else LINGERING_ORANGE
            outline = TRAJECTORY_OUTLINE_COLOR

            if len(positions) < 2:
                last_x, last_y = int(positions[-1][0]), int(positions[-1][1])
                cv2.circle(frame, (last_x, last_y), 5, outline, -1, cv2.LINE_AA)
                cv2.circle(frame, (last_x, last_y), 3, base_color, -1, cv2.LINE_AA)
                continue

            count = len(positions)
            denom = max(count - 1, 1)
            for index in range(1, count):
                point_a = (int(positions[index - 1][0]), int(positions[index - 1][1]))
                point_b = (int(positions[index][0]), int(positions[index][1]))
                recency = index / float(denom)
                dim = tuple(int(0.42 * channel) for channel in base_color)
                color = tuple(
                    int(dim[channel] + (base_color[channel] - dim[channel]) * recency)
                    for channel in range(3)
                )
                thickness = max(2, int(round(2 + 3 * recency)))
                cv2.line(frame, point_a, point_b, outline, thickness + 2, cv2.LINE_AA)
                cv2.line(frame, point_a, point_b, color, thickness, cv2.LINE_AA)
