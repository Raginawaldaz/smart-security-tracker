# 模块：跟踪 ID 归并
# 作用：把同一人被分裂成的多个 track_id 在连续满足几何条件后归并，减少边界处双 ID；
# 外观差异大的不并，已合并的两个 ID 持续分离时自动拆开，避免并肩两人被永久并成一人。

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from smart_security.core.config import IdDedupSettings, get_settings
from smart_security.modules.tracking.constants import PERSON_CLASS_ID, TRACK_ROW_MIN_LEN
from smart_security.modules.tracking.fragment_merge import should_merge_person_pair
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

TrackedObject = list[float]
PersonRow = tuple[float, float, float, float, int, float]


def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
    return (min(int(id_a), int(id_b)), max(int(id_a), int(id_b)))


class TrackIdDeduplicator:
    def __init__(self, params: IdDedupSettings | None = None) -> None:
        params = params or get_settings().id_dedup
        self.id_alias: dict[int, int] = {}
        self._pair_persist: dict[tuple[int, int], int] = {}
        self._split_persist: dict[tuple[int, int], int] = {}
        self._last_seen: dict[int, int] = {}
        self.persist_frames = params.persist_frames
        self.alias_ttl = params.alias_ttl
        self.min_similarity = params.min_similarity
        self.split_frames = params.split_frames
        self.merge_count = 0
        self.split_count = 0

    def canonical_id(self, track_id: int) -> int:
        current = int(track_id)
        seen: list[int] = []
        while current in self.id_alias and self.id_alias[current] != current:
            seen.append(current)
            current = int(self.id_alias[current])
        for previous in seen:
            self.id_alias[previous] = current
        return current

    def _appearance_compatible(
        self, feature_a: np.ndarray | None, feature_b: np.ndarray | None
    ) -> bool:
        if feature_a is None or feature_b is None:
            return True
        norm = float(np.linalg.norm(feature_a) * np.linalg.norm(feature_b))
        if norm <= 0:
            return True
        return float(np.dot(feature_a, feature_b)) / norm >= self.min_similarity

    def apply(
        self,
        tracked_objects: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        frame_count: int,
        features: Mapping[int, np.ndarray] | None = None,
    ) -> list[TrackedObject]:
        features = features or {}
        persons: list[PersonRow] = []
        others: list[TrackedObject] = []
        for obj in tracked_objects:
            if len(obj) < TRACK_ROW_MIN_LEN:
                others.append(list(obj))
                continue
            x1, y1, x2, y2, track_id, conf, cls = obj[:7]
            if int(cls) != PERSON_CLASS_ID:
                others.append(list(obj))
            else:
                persons.append(
                    (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        int(track_id),
                        float(conf),
                    )
                )

        self._update_merges(persons, frame_width, frame_height, features)
        self._release_split_aliases(persons, frame_width, frame_height)
        self._expire_aliases(persons, frame_count)

        groups: dict[int, list[float]] = {}
        for x1, y1, x2, y2, track_id, conf in persons:
            canonical = self.canonical_id(track_id)
            group = groups.get(canonical)
            if group is None:
                groups[canonical] = [x1, y1, x2, y2, conf]
            else:
                group[0] = min(group[0], x1)
                group[1] = min(group[1], y1)
                group[2] = max(group[2], x2)
                group[3] = max(group[3], y2)
                group[4] = max(group[4], conf)

        merged = [
            [
                group[0],
                group[1],
                group[2],
                group[3],
                canonical,
                group[4],
                float(PERSON_CLASS_ID),
            ]
            for canonical, group in groups.items()
        ]
        return merged + list(others)

    def _update_merges(
        self,
        persons: list[PersonRow],
        frame_width: int,
        frame_height: int,
        features: Mapping[int, np.ndarray],
    ) -> None:
        active_pairs: set[tuple[int, int]] = set()
        for i, row_a in enumerate(persons):
            for row_b in persons[i + 1 :]:
                id_a, id_b = row_a[4], row_b[4]
                if self.canonical_id(id_a) == self.canonical_id(id_b):
                    continue
                if not should_merge_person_pair(
                    row_a[:4], row_b[:4], frame_width, frame_height
                ):
                    continue
                if not self._appearance_compatible(
                    features.get(id_a), features.get(id_b)
                ):
                    continue
                key = _pair_key(id_a, id_b)
                active_pairs.add(key)
                self._pair_persist[key] = self._pair_persist.get(key, 0) + 1
                if self._pair_persist[key] >= self.persist_frames:
                    low_id, high_id = key
                    self.id_alias[self.canonical_id(high_id)] = self.canonical_id(
                        low_id
                    )
                    self.merge_count += 1
                    logger.info("Merged track %s into %s", high_id, low_id)

        for key in list(self._pair_persist):
            if key not in active_pairs:
                del self._pair_persist[key]

    def _release_split_aliases(
        self, persons: list[PersonRow], frame_width: int, frame_height: int
    ) -> None:
        by_canonical: dict[int, list[PersonRow]] = {}
        for row in persons:
            by_canonical.setdefault(self.canonical_id(row[4]), []).append(row)

        separated: set[tuple[int, int]] = set()
        for members in by_canonical.values():
            for i, row_a in enumerate(members):
                for row_b in members[i + 1 :]:
                    if should_merge_person_pair(
                        row_a[:4], row_b[:4], frame_width, frame_height
                    ):
                        continue
                    key = _pair_key(row_a[4], row_b[4])
                    separated.add(key)
                    self._split_persist[key] = self._split_persist.get(key, 0) + 1

        for key in list(self._split_persist):
            if key not in separated:
                del self._split_persist[key]
                continue
            if self._split_persist[key] < self.split_frames:
                continue
            _low_id, high_id = key
            self.id_alias.pop(high_id, None)
            del self._split_persist[key]
            self.split_count += 1
            logger.info("Split track %s from %s", high_id, key[0])

    def _expire_aliases(self, persons: list[PersonRow], frame_count: int) -> None:
        for row in persons:
            self._last_seen[row[4]] = frame_count
        for raw_id in list(self.id_alias):
            last = self._last_seen.get(raw_id, frame_count)
            if frame_count - last > self.alias_ttl:
                self.id_alias.pop(raw_id, None)
                self._last_seen.pop(raw_id, None)
