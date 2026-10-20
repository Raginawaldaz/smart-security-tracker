# 模块：报警常量
# 作用：抽出逗留进出缓冲与展示阈值，避免主流程出现裸数字。

from enum import Enum

ZONE_LEAVE_MIN_LINGER_SECONDS = 3.0
ZONE_UNDETECTED_TIMEOUT_SECONDS = 5.0
LINGER_LABEL_MIN_SECONDS = 3.0


class AlarmType(str, Enum):
    INTRUSION = "intrusion"
    LINGERING = "lingering"


class ZoneShape(str, Enum):
    RECTANGLE = "rectangle"


class LingerState(str, Enum):
    OUTSIDE = "outside"
    ENTERING = "entering"
    INSIDE = "inside"
    LEAVING = "leaving"
