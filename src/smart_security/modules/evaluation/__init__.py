# 模块：离线评测
# 作用：用标注视频量化报警准确率与 ID 稳定性，判断瓶颈在算法还是模型。

from smart_security.modules.evaluation.metrics import (
    AlarmHit,
    AlarmMetrics,
    LingerEvent,
    load_events,
    score_alarms,
    zone_ids_per_event,
)

__all__ = [
    "AlarmHit",
    "AlarmMetrics",
    "LingerEvent",
    "load_events",
    "score_alarms",
    "zone_ids_per_event",
]
