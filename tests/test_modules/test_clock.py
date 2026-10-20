# 模块：时间源测试
# 作用：视频时间戳缺失或回退时仍单调递增；注入时钟帧内取值稳定。

from __future__ import annotations

import pytest

from smart_security.modules.tracking.clock import FrameClock, VideoTimeline


def test_timeline_uses_container_timestamps() -> None:
    timeline = VideoTimeline(fps=25.0)
    assert timeline.next(0.0) == 0.0
    assert timeline.next(40.0) == pytest.approx(0.04)
    assert timeline.next(80.0) == pytest.approx(0.08)


def test_timeline_falls_back_to_frame_rate_when_missing() -> None:
    timeline = VideoTimeline(fps=10.0)
    stamps = [timeline.next(0.0) for _ in range(3)]
    assert stamps == pytest.approx([0.0, 0.1, 0.2])


def test_timeline_never_goes_backwards() -> None:
    timeline = VideoTimeline(fps=10.0)
    timeline.next(1000.0)
    assert timeline.next(500.0) == pytest.approx(1.1)


def test_frame_clock_holds_value_between_ticks() -> None:
    ticks = iter([1.0, 2.0, 3.0])
    clock = FrameClock(lambda: next(ticks))
    assert clock() == 1.0
    assert clock.tick() == 2.0
    assert clock() == 2.0
    assert clock.tick(42.0) == 42.0
    assert clock() == 42.0
