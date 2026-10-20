# 模块：通用工具
# 作用：暴露日志初始化与启动补丁。

from smart_security.utils.logger import get_logger, setup_logging
from smart_security.utils.patches import apply_all

__all__ = ["setup_logging", "get_logger", "apply_all"]
