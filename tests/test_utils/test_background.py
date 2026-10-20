# 模块：后台写盘与采集线程测试
# 作用：确认任务按序执行、单任务异常不终止线程、满队列时丢弃而非阻塞，采集只交付最新帧。

from __future__ import annotations

import threading

import numpy as np

from smart_security.modules.tracking.capture import LatestFrameCapture
from smart_security.utils.background import BackgroundWriter


def test_tasks_run_in_submission_order() -> None:
    writer = BackgroundWriter()
    seen: list[int] = []
    for value in range(5):
        writer.submit(seen.append, value)
    writer.flush()
    writer.close()
    assert seen == [0, 1, 2, 3, 4]


def test_failing_task_does_not_stop_worker() -> None:
    writer = BackgroundWriter()
    seen: list[str] = []

    def boom() -> None:
        raise OSError("disk full")

    writer.submit(boom)
    writer.submit(seen.append, "after")
    writer.flush()
    writer.close()
    assert seen == ["after"]


def test_full_queue_drops_instead_of_blocking() -> None:
    writer = BackgroundWriter(max_pending=1)
    release = threading.Event()
    started = threading.Event()

    def block() -> None:
        started.set()
        release.wait(5.0)

    assert writer.submit(block)
    started.wait(5.0)
    assert writer.submit(lambda: None)
    assert not writer.submit(lambda: None)
    release.set()
    writer.close()
    assert not writer.submit(lambda: None)


class _FakeCamera:
    def __init__(self, frames: int) -> None:
        self._remaining = frames
        self._value = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._remaining == 0:
            return False, None
        self._remaining -= 1
        self._value += 1
        return True, np.full((2, 2), self._value, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_latest_frame_capture_ends_with_source() -> None:
    camera = _FakeCamera(frames=3)
    capture = LatestFrameCapture(camera, read_timeout=1.0)
    values: list[int] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        values.append(int(frame[0, 0]))
    capture.release()
    assert values
    assert values == sorted(values)
    assert values[-1] == 3
    assert camera.released
