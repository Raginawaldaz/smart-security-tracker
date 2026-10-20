# 模块：跟踪断链重关联
# 作用：遮挡或漏检后跟踪器常给同一人分配新 ID，逗留计时随之清零；
# 新 ID 出现时与最近消失的旧轨迹按足点距离 + ReID 外观匹配，命中则沿用旧 ID，
# 下游的区内判定、逗留计时和轨迹都按旧 ID 继续累计。

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from smart_security.core.config import RelinkSettings, get_settings
from smart_security.modules.tracking.constants import PERSON_CLASS_ID, TRACK_ROW_MIN_LEN
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

TrackedRow = list[float]
Box = tuple[float, float, float, float]

# 缺少外观特征时只允许近距离接续，并给一个低于正常外观匹配的固定分。
NO_APPEARANCE_SCORE = 0.5
DISTANCE_PENALTY = 0.2


@dataclass
class _TrackMemory:
    box: Box
    last_time: float
    feature: np.ndarray | None


def _foot(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, box[3]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class TrackRelinker:
    def __init__(self, params: RelinkSettings | None = None) -> None:
        self.params = params or get_settings().relink
        self._alias: dict[int, int] = {}
        self._raw_last_seen: dict[int, float] = {}
        self._memory: dict[int, _TrackMemory] = {}
        self.relink_count = 0

    def canonical_id(self, raw_id: int) -> int:
        return self._alias.get(int(raw_id), int(raw_id))

    def apply(
        self,
        tracked_objects: Sequence[Sequence[float]],
        now: float,
        features: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[list[TrackedRow], dict[int, np.ndarray]]:
        """返回改写 ID 后的行，以及按输出 ID 索引的外观特征。"""
        features = features or {}
        rows = [list(obj) for obj in tracked_objects]
        person_rows = [
            row
            for row in rows
            if len(row) >= TRACK_ROW_MIN_LEN and int(row[6]) == PERSON_CLASS_ID
        ]
        boxes = {int(row[4]): (row[0], row[1], row[2], row[3]) for row in person_rows}

        assignment = self._assign(boxes, features, now)
        for row in person_rows:
            row[4] = float(assignment[int(row[4])])

        out_features: dict[int, np.ndarray] = {}
        for raw_id, canonical in assignment.items():
            self._raw_last_seen[raw_id] = now
            feature = features.get(raw_id)
            self._remember(canonical, boxes[raw_id], now, feature)
            if feature is not None:
                out_features[canonical] = feature
        self._expire(now)
        return rows, out_features

    def _assign(
        self, boxes: dict[int, Box], features: Mapping[int, np.ndarray], now: float
    ) -> dict[int, int]:
        assignment: dict[int, int] = {}
        claimed: set[int] = set()
        known = [raw for raw in boxes if raw in self._raw_last_seen]
        # 原始 ID 优先占用自己的 canonical：跟踪器找回旧轨迹时，此前重关联出的新 ID 退回自身。
        known.sort(key=lambda raw: (self.canonical_id(raw) != raw, raw))
        for raw in known:
            canonical = self.canonical_id(raw)
            if canonical in claimed:
                self._alias.pop(raw, None)
                canonical = raw
            assignment[raw] = canonical
            claimed.add(canonical)

        new_ids = sorted(raw for raw in boxes if raw not in self._raw_last_seen)
        if self.params.enabled and new_ids:
            for raw, canonical in self._match_new(
                new_ids, boxes, features, claimed, now
            ):
                self._alias[raw] = canonical
                assignment[raw] = canonical
                claimed.add(canonical)
                self.relink_count += 1
                logger.info("Relinked track %s -> %s", raw, canonical)
        for raw in new_ids:
            assignment.setdefault(raw, raw)
        return assignment

    def _match_new(
        self,
        new_ids: list[int],
        boxes: dict[int, Box],
        features: Mapping[int, np.ndarray],
        claimed: set[int],
        now: float,
    ) -> list[tuple[int, int]]:
        lost = {
            canonical: memory
            for canonical, memory in self._memory.items()
            if canonical not in claimed
            and now - memory.last_time <= self.params.max_gap_seconds
        }
        candidates: list[tuple[float, int, int]] = []
        for raw in new_ids:
            for canonical, memory in lost.items():
                score = self._score(boxes[raw], features.get(raw), memory, now)
                if score is not None:
                    candidates.append((score, raw, canonical))
        candidates.sort(reverse=True)
        used_raw: set[int] = set()
        used_canonical: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _score, raw, canonical in candidates:
            if raw in used_raw or canonical in used_canonical:
                continue
            used_raw.add(raw)
            used_canonical.add(canonical)
            matches.append((raw, canonical))
        return matches

    def _score(
        self,
        box: Box,
        feature: np.ndarray | None,
        memory: _TrackMemory,
        now: float,
    ) -> float | None:
        gap = max(now - memory.last_time, 0.0)
        height = max(box[3] - box[1], memory.box[3] - memory.box[1], 1.0)
        foot_x, foot_y = _foot(box)
        last_x, last_y = _foot(memory.box)
        distance = float(np.hypot(foot_x - last_x, foot_y - last_y)) / height
        limit = self.params.max_dist_ratio + self.params.max_speed_ratio * gap
        if distance > limit:
            return None
        if feature is None or memory.feature is None:
            if distance > self.params.max_dist_ratio:
                return None
            return NO_APPEARANCE_SCORE - DISTANCE_PENALTY * distance / limit
        similarity = float(np.dot(_normalize(feature), memory.feature))
        if similarity < self.params.min_similarity:
            return None
        return similarity - DISTANCE_PENALTY * distance / limit

    def _remember(
        self, canonical: int, box: Box, now: float, feature: np.ndarray | None
    ) -> None:
        memory = self._memory.get(canonical)
        normalized = (
            _normalize(np.asarray(feature, dtype=np.float32))
            if feature is not None
            else None
        )
        if memory is None:
            self._memory[canonical] = _TrackMemory(box, now, normalized)
            return
        memory.box = box
        memory.last_time = now
        if normalized is None:
            return
        if memory.feature is None or memory.feature.shape != normalized.shape:
            memory.feature = normalized
        else:
            ema = self.params.feature_ema
            memory.feature = _normalize(ema * memory.feature + (1.0 - ema) * normalized)

    def _expire(self, now: float) -> None:
        horizon = self.params.max_gap_seconds
        for canonical in [
            key
            for key, memory in self._memory.items()
            if now - memory.last_time > horizon
        ]:
            del self._memory[canonical]
        for raw in [
            key for key, seen in self._raw_last_seen.items() if now - seen > horizon
        ]:
            del self._raw_last_seen[raw]
            self._alias.pop(raw, None)
