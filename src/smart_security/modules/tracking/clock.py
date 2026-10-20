# 模块：帧时间源
# 作用：实时流用墙钟、视频文件用帧时间戳，让逗留时长与处理速度无关。

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

DEFAULT_VIDEO_FPS = 30.0
MSEC_PER_SECOND = 1000.0


class FrameClock:
    """逗留检测等模块注入的时间函数；每帧由主循环 tick 一次，帧内取值保持一致。"""

    def __init__(self, time_fn: Callable[[], float] = time.time) -> None:
        self._time_fn = time_fn
        self._now = time_fn()

    def tick(self, timestamp: float | None = None) -> float:
        self._now = self._time_fn() if timestamp is None else float(timestamp)
        return self._now

    def __call__(self) -> float:
        return self._now


class VideoTimeline:
    """把 CAP_PROP_POS_MSEC 转成单调递增秒数；部分后端返回 0 或回退时按帧率补齐。"""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_VIDEO_FPS
        self._index = 0
        self._last: float | None = None

    def next(self, pos_msec: float) -> float:
        timestamp = (
            float(pos_msec) / MSEC_PER_SECOND
            if pos_msec and pos_msec > 0
            else self._index / self.fps
        )
        if self._last is not None and timestamp <= self._last:
            timestamp = self._last + 1.0 / self.fps
        self._last = timestamp
        self._index += 1
        return timestamp


def is_file_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).is_file()
