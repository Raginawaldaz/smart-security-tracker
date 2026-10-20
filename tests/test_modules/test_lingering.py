# 模块：逗留状态机测试
# 作用：锁定进出帧缓冲、超时报警与离开清理，避免回归成即时进出。

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.lingering import LingeringDetector


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
        zone_in_frames=4,
        zone_out_frames=10,
    )


def test_requires_in_frames_before_register(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(3):
        detector.check([_person(3)], frame, _always_in)
    assert 3 not in detector.lingering_targets
    detector.check([_person(3)], frame, _always_in)
    assert 3 in detector.lingering_targets


def test_alert_after_threshold_and_cleanup_after_out_frames(tmp_path: Path) -> None:
    clock = Clock()
    detector = _detector(tmp_path, clock)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(4):
        detector.check([_person(8)], frame, _always_in)
    clock.now += 15.0
    alerts = detector.check([_person(8)], frame, _always_in)
    assert len(alerts) == 1
    assert alerts[0].track_id == 8
    assert detector.lingering_targets[8]["alert_count"] == 1

    for _ in range(10):
        detector.check([_person(8)], frame, _always_out)
    assert 8 not in detector.lingering_targets
