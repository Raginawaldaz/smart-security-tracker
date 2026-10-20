# 模块：报警数据结构
# 作用：统一定义逗留/入侵报警的字段，避免 dict 键名在各模块漂移。

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from smart_security.modules.alarm.constants import AlarmType


@dataclass
class AlertRecord:
    timestamp: str
    track_id: int
    linger_time: float
    screenshot: str
    type: str = AlarmType.LINGERING.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlarmEvent:
    timestamp: str
    zone_id: int
    zone_name: str
    track_id: int
    alarm_type: str
    linger_time: float | None = None
    bbox: list[float] | None = None
    screenshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
