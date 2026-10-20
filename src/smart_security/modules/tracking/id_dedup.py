# 模块：跟踪 ID 归并
# 作用：把同一人被分裂成的多个 track_id 在连续满足几何条件后归并，减少边界处双 ID。

from __future__ import annotations

from collections.abc import Sequence

from smart_security.core.config import get_settings
from smart_security.modules.tracking.constants import PERSON_CLASS_ID, TRACK_ROW_MIN_LEN
from smart_security.modules.tracking.fragment_merge import should_merge_person_pair

TrackedObject = list[float]


class TrackIdDeduplicator:
    def __init__(self) -> None:
        settings = get_settings()
        self.id_alias: dict[int, int] = {}
        self._pair_persist: dict[tuple[int, int], int] = {}
        self._last_seen: dict[int, int] = {}
        self.persist_frames = settings.id_dedup.persist_frames
        self.alias_ttl = settings.id_dedup.alias_ttl

    def canonical_id(self, track_id: int) -> int:
        current = int(track_id)
        seen: list[int] = []
        while current in self.id_alias and self.id_alias[current] != current:
            seen.append(current)
            current = int(self.id_alias[current])
        for previous in seen:
            self.id_alias[previous] = current
        return current

    def apply(
        self,
        tracked_objects: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        frame_count: int,
    ) -> list[TrackedObject]:
        persons: list[list[float]] = []
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
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        int(track_id),
                        float(conf),
                    ]
                )

        active_pairs: set[tuple[int, int]] = set()
        person_count = len(persons)
        for i in range(person_count):
            x1_a, y1_a, x2_a, y2_a, id_a, _conf_a = persons[i]
            for j in range(i + 1, person_count):
                x1_b, y1_b, x2_b, y2_b, id_b, _conf_b = persons[j]
                if should_merge_person_pair(
                    (x1_a, y1_a, x2_a, y2_a),
                    (x1_b, y1_b, x2_b, y2_b),
                    frame_width,
                    frame_height,
                ):
                    key = (min(int(id_a), int(id_b)), max(int(id_a), int(id_b)))
                    active_pairs.add(key)
                    self._pair_persist[key] = self._pair_persist.get(key, 0) + 1
                    if self._pair_persist[key] >= self.persist_frames:
                        low_id, high_id = key
                        self.id_alias[self.canonical_id(high_id)] = self.canonical_id(
                            low_id
                        )

        for key in list(self._pair_persist):
            if key not in active_pairs:
                del self._pair_persist[key]

        for _x1, _y1, _x2, _y2, track_id, _conf in persons:
            self._last_seen[int(track_id)] = frame_count
        for raw_id in list(self.id_alias):
            last = self._last_seen.get(raw_id, frame_count)
            if frame_count - last > self.alias_ttl:
                self.id_alias.pop(raw_id, None)
                self._last_seen.pop(raw_id, None)

        groups: dict[int, list[float]] = {}
        for x1, y1, x2, y2, track_id, conf in persons:
            canonical = self.canonical_id(int(track_id))
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
