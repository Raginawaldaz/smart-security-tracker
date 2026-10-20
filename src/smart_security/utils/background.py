# 模块：后台写盘线程
# 作用：把截图、报警 JSON、预览帧等磁盘 IO 移出主循环，避免写盘卡顿拖慢检测帧率。

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PENDING = 64
DEFAULT_CLOSE_TIMEOUT = 5.0

Task = tuple[Callable[..., Any], tuple[Any, ...]]


class BackgroundWriter:
    """单个工作线程按提交顺序执行任务，保证报警 JSON 追加不乱序。"""

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._queue: queue.Queue[Task | None] = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="background-writer", daemon=True
        )
        self._thread.start()

    def submit(self, fn: Callable[..., Any], *args: Any) -> bool:
        """队列满时丢弃并返回 False：宁可少存一张图，也不阻塞实时检测。"""
        if self._closed:
            return False
        try:
            self._queue.put_nowait((fn, args))
        except queue.Full:
            logger.warning("Background writer queue full, dropping %s", _name(fn))
            return False
        return True

    def flush(self) -> None:
        self._queue.join()

    def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("Background writer did not finish within %.1fs", timeout)

    def _loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                fn, args = task
                try:
                    fn(*args)
                except Exception:  # 工作线程不能因单个任务失败而退出
                    logger.exception("Background task %s failed", _name(fn))
            finally:
                self._queue.task_done()


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", repr(fn))
