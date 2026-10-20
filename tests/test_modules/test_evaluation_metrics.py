# 模块：评测指标测试
# 作用：锁定报警与标注事件的对齐规则，以及每个事件内区内 ID 数的统计。

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_security.modules.evaluation.metrics import (
    AlarmHit,
    LingerEvent,
    load_events,
    score_alarms,
    zone_ids_per_event,
)

THRESHOLD = 15.0


def test_alarm_inside_window_counts_once_per_event() -> None:
    events = [LingerEvent(10.0, 40.0)]
    alarms = [AlarmHit(25.5, 1), AlarmHit(40.5, 1)]
    metrics = score_alarms(events, alarms, THRESHOLD)
    assert metrics.detected_events == 1
    assert metrics.true_alarms == 2
    assert metrics.false_alarms == 0
    assert metrics.recall == 1.0
    assert metrics.mean_latency == pytest.approx(0.5)


def test_short_event_alarm_is_false_positive() -> None:
    events = [LingerEvent(0.0, 8.0)]
    metrics = score_alarms(events, [AlarmHit(7.0, 3)], THRESHOLD)
    assert metrics.expected_events == 0
    assert metrics.false_alarms == 1
    assert metrics.precision == 0.0
    assert metrics.recall is None


def test_missed_event_lowers_recall() -> None:
    events = [LingerEvent(0.0, 30.0), LingerEvent(100.0, 130.0)]
    metrics = score_alarms(events, [AlarmHit(15.2, 1)], THRESHOLD)
    assert metrics.recall == 0.5
    assert metrics.precision == 1.0


def test_zone_ids_per_event_counts_id_switches() -> None:
    events = [LingerEvent(0.0, 10.0), LingerEvent(20.0, 30.0)]
    observations = [(1.0, [1]), (5.0, [4]), (9.0, [4]), (25.0, [9]), (50.0, [7])]
    assert zone_ids_per_event(observations, events) == [2, 1]


def test_load_events_sorts_by_start(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"events": [{"start": 30, "end": 50}, {"start": 1, "end": 20}]}),
        encoding="utf-8",
    )
    assert load_events(path) == [LingerEvent(1.0, 20.0), LingerEvent(30.0, 50.0)]
