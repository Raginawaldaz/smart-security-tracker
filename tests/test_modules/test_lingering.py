# 模块：逗留状态机测试
# 作用：锁定按秒计的进出缓冲、进入时间回溯、超时报警与离开清理，避免回归成即时进出。

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.lingering import LingeringDetector

FRAME_STEP = 0.1


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _person(track_id: int) -> list[float]:
    return [10.0, 20.0, 80.0, 200.0, float(track_id), 0.9, 0.0]


def _always_in(_track_id: int) -> bool:
    return True


def _always_out(_track_id: int) -> bool:
    return False


def _detector(tmp_path: Path, clock: Clock) -> LingeringDetector:
    store = AlertStore(
        screenshots_dir=tmp_path / "shots",
        alerts_file=tmp_path / "alerts.json",
    )
    return LingeringDetector(
        alert_store=store,
        time_fn=clock,
        lingering_threshold=15.0,
        zone_in_seconds=0.25,
        zone_out_seconds=0.95,
        undetected_timeout=5.0,
    )


def _step(
    detector: LingeringDetector, clock: Clock, rows: list, in_zone=_always_in
) -> list:
    alerts = detector.check(rows, np.zeros((64, 64, 3), dtype=np.uint8), in_zone)
    clock.now += FRAME_STEP
    return alerts


def test_requires_in_seconds_before_register(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    for _ in range(3):
        _step(detector, clock, [_person(3)])
    assert 3 not in detector.lingering_targets
    _step(detector, clock, [_person(3)])
    assert 3 in detector.lingering_targets


def test_enter_time_backdated_to_first_in_zone_frame(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    first_seen = clock.now
    for _ in range(4):
        _step(detector, clock, [_person(5)])
    assert detector.lingering_targets[5]["enter_time"] == first_seen


def test_alert_after_threshold_and_cleanup_after_out_seconds(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    start = clock.now
    for _ in range(4):
        _step(detector, clock, [_person(8)])
    clock.now = start + 15.0
    alerts = _step(detector, clock, [_person(8)])
    assert len(alerts) == 1
    assert alerts[0].track_id == 8
    assert detector.lingering_targets[8]["alert_count"] == 1

    for _ in range(10):
        _step(detector, clock, [_person(8)], _always_out)
    assert 8 in detector.lingering_targets
    _step(detector, clock, [_person(8)], _always_out)
    assert 8 not in detector.lingering_targets


def test_short_detection_gap_keeps_entering_progress(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    first_seen = clock.now
    _step(detector, clock, [_person(4)])
    _step(detector, clock, [])
    _step(detector, clock, [_person(4)])
    _step(detector, clock, [_person(4)])
    assert detector.lingering_targets[4]["enter_time"] == first_seen


def test_undetected_target_removed_after_timeout_without_detections(
    tmp_path: Path,
) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    for _ in range(4):
        _step(detector, clock, [_person(6)])
    assert 6 in detector.lingering_targets
    clock.now += 5.5
    _step(detector, clock, [])
    assert 6 not in detector.lingering_targets
