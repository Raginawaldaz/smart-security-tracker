# 模块：项目自定义异常
# 作用：用明确异常类型替代返回 None/False，保留失败语义供入口统一处理。


class SmartSecurityError(Exception):
    """项目内所有不可恢复错误的基类。"""


class ConfigError(SmartSecurityError):
    """配置文件缺失、损坏或字段类型不合法。"""


class DatabaseError(SmartSecurityError):
    """SQLite 读写失败。"""


class ModelLoadError(SmartSecurityError):
    """YOLO / ReID / 跟踪器权重加载失败。"""


class CameraError(SmartSecurityError):
    """摄像头打不开或读帧失败。"""


class ZoneConfigError(SmartSecurityError):
    """警戒区配置无法持久化。"""


class AlertStoreError(SmartSecurityError):
    """报警截图或 JSON 落盘失败。"""
