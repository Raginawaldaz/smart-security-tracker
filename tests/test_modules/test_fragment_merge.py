# 模块：人员碎片合并测试
# 作用：锁定竖切半框应合并、横向远离的两人不应合并。

from __future__ import annotations

from smart_security.core.config import FragmentMergeSettings
from smart_security.modules.tracking.fragment_merge import (
    merge_fragment_person_detections,
    should_merge_person_pair,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PARAMS = FragmentMergeSettings()


def test_vertical_split_halves_should_merge() -> None:
    left_half = (10.0, 20.0, 55.0, 220.0)
    right_half = (55.0, 20.0, 100.0, 220.0)
    assert should_merge_person_pair(
        left_half, right_half, FRAME_WIDTH, FRAME_HEIGHT, PARAMS
    )


def test_far_apart_people_should_not_merge() -> None:
    person_a = (10.0, 20.0, 80.0, 220.0)
    person_b = (300.0, 20.0, 370.0, 220.0)
    assert not should_merge_person_pair(
        person_a, person_b, FRAME_WIDTH, FRAME_HEIGHT, PARAMS
    )


def test_merge_reduces_person_count() -> None:
    detections = [
        [10.0, 20.0, 55.0, 220.0, 0.9, 0.0],
        [55.0, 20.0, 100.0, 220.0, 0.8, 0.0],
        [400.0, 30.0, 500.0, 90.0, 0.7, 2.0],
    ]
    merged = merge_fragment_person_detections(
        detections, FRAME_WIDTH, FRAME_HEIGHT, PARAMS
    )
    person_boxes = [row for row in merged if int(row[5]) == 0]
    other_boxes = [row for row in merged if int(row[5]) != 0]
    assert len(person_boxes) == 1
    assert person_boxes[0][0] == 10.0
    assert person_boxes[0][2] == 100.0
    assert person_boxes[0][4] == 0.9
    assert len(other_boxes) == 1
