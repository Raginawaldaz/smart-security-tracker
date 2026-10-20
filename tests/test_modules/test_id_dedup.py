# 模块：跟踪 ID 归并测试
# 作用：连续满足竖切几何条件后两个 track_id 应收成一个。

from __future__ import annotations

from smart_security.modules.tracking.id_dedup import TrackIdDeduplicator

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _split_person(track_id_left: int, track_id_right: int) -> list[list[float]]:
    return [
        [10.0, 20.0, 55.0, 220.0, float(track_id_left), 0.9, 0.0],
        [55.0, 20.0, 100.0, 220.0, float(track_id_right), 0.8, 0.0],
    ]


def test_ids_merge_after_persist_frames() -> None:
    dedup = TrackIdDeduplicator()
    persist = dedup.persist_frames
    tracked = None
    for frame_count in range(1, persist + 1):
        tracked = dedup.apply(
            _split_person(11, 22), FRAME_WIDTH, FRAME_HEIGHT, frame_count
        )
    persons = [row for row in tracked or [] if int(row[6]) == 0]
    assert len(persons) == 1
    assert int(persons[0][4]) == 11
    assert persons[0][0] == 10.0
    assert persons[0][2] == 100.0


def test_non_person_rows_pass_through() -> None:
    dedup = TrackIdDeduplicator()
    rows = [[1.0, 2.0, 3.0, 4.0, 9.0, 0.5, 2.0]]
    result = dedup.apply(rows, FRAME_WIDTH, FRAME_HEIGHT, frame_count=1)
    assert result == rows
