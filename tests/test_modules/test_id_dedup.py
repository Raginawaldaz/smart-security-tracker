# 模块：跟踪 ID 归并测试
# 作用：连续满足竖切几何条件后两个 track_id 应收成一个；外观不符不并，持续分离后拆开。

from __future__ import annotations

import numpy as np

from smart_security.core.config import IdDedupSettings
from smart_security.modules.tracking.id_dedup import TrackIdDeduplicator

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _split_person(track_id_left: int, track_id_right: int) -> list[list[float]]:
    return [
        [10.0, 20.0, 55.0, 220.0, float(track_id_left), 0.9, 0.0],
        [55.0, 20.0, 100.0, 220.0, float(track_id_right), 0.8, 0.0],
    ]


def _apart(track_id_left: int, track_id_right: int) -> list[list[float]]:
    return [
        [10.0, 20.0, 80.0, 220.0, float(track_id_left), 0.9, 0.0],
        [400.0, 20.0, 470.0, 220.0, float(track_id_right), 0.8, 0.0],
    ]


def _dedup() -> TrackIdDeduplicator:
    return TrackIdDeduplicator(
        IdDedupSettings(
            persist_frames=3, alias_ttl=150, min_similarity=0.5, split_frames=4
        )
    )


def _persons(rows: list[list[float]]) -> list[list[float]]:
    return [row for row in rows if int(row[6]) == 0]


def test_ids_merge_after_persist_frames() -> None:
    dedup = TrackIdDeduplicator()
    persist = dedup.persist_frames
    tracked = None
    for frame_count in range(1, persist + 1):
        tracked = dedup.apply(
            _split_person(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count
        )
    persons = _persons(tracked or [])
    assert len(persons) == 1
    assert int(persons[0][4]) == 11
    assert persons[0][0] == 10.0
    assert persons[0][2] == 100.0
    assert dedup.merge_count == 1


def test_non_person_rows_pass_through() -> None:
    dedup = TrackIdDeduplicator()
    rows = [[1.0, 2.0, 3.0, 4.0, 9.0, 0.5, 2.0]]
    result = dedup.apply(rows, FRAME_WIDTH, FRAME_HEIGHT, frame_count=1)
    assert result == rows


def test_dissimilar_appearance_blocks_merge() -> None:
    dedup = _dedup()
    features = {
        11: np.array([1.0, 0.0], dtype=np.float32),
        22: np.array([0.0, 1.0], dtype=np.float32),
    }
    tracked: list[list[float]] = []
    for frame_count in range(1, 6):
        tracked = dedup.apply(
            _split_person(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count, features
        )
    assert len(_persons(tracked)) == 2


def test_merged_ids_split_after_separating() -> None:
    dedup = _dedup()
    frame_count = 0
    for _ in range(3):
        frame_count += 1
        dedup.apply(_split_person(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count)
    assert dedup.canonical_id(22) == 11

    tracked: list[list[float]] = []
    for _ in range(3):
        frame_count += 1
        tracked = dedup.apply(_apart(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count)
    assert len(_persons(tracked)) == 1

    frame_count += 1
    tracked = dedup.apply(_apart(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count)
    assert sorted(int(row[4]) for row in _persons(tracked)) == [11, 22]
    assert dedup.split_count == 1
