# 模块：实时采集线程
# 作用：摄像头在独立线程持续读帧、只保留最新一帧，避免推理慢于帧率时驱动缓冲堆积造成延迟。
# 视频文件不能丢帧，不使用本类。

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

DEFAULT_READ_TIMEOUT = 5.0
DEFAULT_JOIN_TIMEOUT = 2.0


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class LatestFrameCapture:
    def __init__(
        self, source: FrameSource, read_timeout: float = DEFAULT_READ_TIMEOUT
    ) -> None:
        self._source = source
        self._read_timeout = read_timeout
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._consumed = 0
        self._stopped = False
        self._thread = threading.Thread(
            target=self._loop, name="frame-capture", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            ok, frame = self._source.read()
            with self._condition:
                if not ok:
                    self._stopped = True
                    self._condition.notify_all()
                    return
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """阻塞到出现比上次更新的帧；源结束或超时返回 (False, None)。"""
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != self._consumed or self._stopped,
                timeout=self._read_timeout,
            )
            if self._sequence == self._consumed:
                return False, None
            self._consumed = self._sequence
            return True, self._frame

    def release(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(DEFAULT_JOIN_TIMEOUT)
        self._source.release()
