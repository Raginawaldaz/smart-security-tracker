# 模块：多目标跟踪器工厂
# 作用：按配置名创建 boxmot 跟踪器，并复用其 ReID 模型为每帧检测只算一次外观特征，
# 同一份特征同时喂给跟踪器、断链重关联与 ID 归并。

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smart_security.core.config import Settings, get_settings
from smart_security.core.exceptions import ModelLoadError
from smart_security.utils.logger import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

SUPPORTED_TRACKERS: tuple[str, ...] = (
    "strongsort",
    "botsort",
    "deepocsort",
    "boosttrack",
)
REID_BACKEND_ATTRIBUTES: tuple[str, ...] = ("model", "reid_model")


class AppearanceEncoder:
    """包装 boxmot ReID 后端；输出按行 L2 归一化的特征。"""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def encode(self, xyxy: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
        if xyxy.size == 0:
            return None
        features = np.asarray(self._backend.get_features(xyxy, frame), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != xyxy.shape[0]:
            return None
        return features


def _reid_weights(cfg: Settings) -> Path | None:
    reid_path = cfg.reid_weights_path
    return reid_path if reid_path.exists() else None


def build_tracker(
    name: str, device: torch.device, settings: Settings | None = None
) -> Any:
    cfg = settings or get_settings()
    key = name.strip().lower()
    if key not in SUPPORTED_TRACKERS:
        raise ModelLoadError(
            f"Unknown tracker '{name}', choose from {', '.join(SUPPORTED_TRACKERS)}"
        )
    try:
        import boxmot
    except ImportError as exc:
        raise ModelLoadError(
            "boxmot is not installed; run pip install -r requirements.txt"
        ) from exc

    common: dict[str, Any] = {
        "reid_weights": _reid_weights(cfg),
        "device": str(device),
        "half": device.type == "cuda",
        "max_age": cfg.tracking.max_age,
        "min_hits": cfg.tracking.min_hits,
    }
    detection_conf = cfg.detection.confidence_threshold
    builders: dict[str, Callable[[], Any]] = {
        "strongsort": lambda: boxmot.StrongSort(
            **common, max_iou_dist=cfg.tracking.max_iou_dist
        ),
        "botsort": lambda: boxmot.BotSort(
            **common,
            track_buffer=cfg.tracking.max_age,
            frame_rate=cfg.camera.fps,
            with_reid=True,
        ),
        "deepocsort": lambda: boxmot.DeepOcSort(**common, det_thresh=detection_conf),
        "boosttrack": lambda: boxmot.BoostTrack(
            **common, det_thresh=detection_conf, with_reid=True
        ),
    }
    try:
        tracker = builders[key]()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"Tracker initialization failed: {key}") from exc
    logger.info("Tracker %s initialized on %s", key, device)
    return tracker


def build_appearance_encoder(
    tracker: Any, device: torch.device, settings: Settings | None = None
) -> AppearanceEncoder | None:
    """优先复用跟踪器内置的 ReID 后端；拿不到时单独加载，失败则退化为无外观特征。"""
    for attribute in REID_BACKEND_ATTRIBUTES:
        backend = getattr(tracker, attribute, None)
        if backend is not None and hasattr(backend, "get_features"):
            return AppearanceEncoder(backend)

    cfg = settings or get_settings()
    try:
        from boxmot.appearance.reid.auto_backend import ReidAutoBackend

        backend = ReidAutoBackend(
            weights=_reid_weights(cfg),
            device=str(device),
            half=device.type == "cuda",
        ).model
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "ReID encoder unavailable, relink/dedup lose appearance: %s", exc
        )
        return None
    return AppearanceEncoder(backend)
