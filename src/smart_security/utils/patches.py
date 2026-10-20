# 模块：PyTorch 权重加载补丁
# 作用：在 torch>=2.6 下注册 YOLO checkpoint 常用 safe globals，避免反序列化被拒。

from __future__ import annotations

import importlib
from collections.abc import Sequence

import torch

from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

CORE_SAFE_GLOBAL_PATHS: tuple[str, ...] = (
    "torch._utils._rebuild_tensor_v2",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.conv.Conv2d",
    "collections.OrderedDict",
)

YOLO_SAFE_GLOBAL_PATHS: tuple[str, ...] = (
    "ultralytics.nn.tasks.DetectionModel",
    "ultralytics.nn.tasks.SegmentationModel",
    "ultralytics.nn.tasks.PoseModel",
    "ultralytics.nn.modules.Conv",
    "ultralytics.nn.modules.Bottleneck",
    "ultralytics.nn.modules.C2f",
    "ultralytics.nn.modules.SPPF",
)


def _resolve_obj(dotted_path: str) -> object:
    module_name, _, attr_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _register_safe_globals(paths: Sequence[str]) -> int:
    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe is None:
        logger.warning("torch.serialization.add_safe_globals unavailable, skip")
        return 0

    registered = 0
    for path in paths:
        try:
            obj = _resolve_obj(path)
            add_safe([obj])
            registered += 1
        except (ImportError, AttributeError, ValueError) as exc:
            # 不同 torch/ultralytics 版本符号名不一致，缺失时跳过以免阻断启动。
            logger.warning("skip safe global %s: %s", path, exc)
    return registered


def apply_all() -> bool:
    try:
        count_core = _register_safe_globals(CORE_SAFE_GLOBAL_PATHS)
        count_yolo = _register_safe_globals(YOLO_SAFE_GLOBAL_PATHS)
        logger.info("applied safe globals: core=%s, yolo=%s", count_core, count_yolo)
        return True
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.exception("failed to apply torch safe-global patches: %s", exc)
        return False
