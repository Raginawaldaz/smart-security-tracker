# 模块：YOLO 检测与 StrongSORT 初始化
# 作用：隔离模型加载失败点，把权重路径和设备选择从主循环中拆出。

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smart_security.core.config import Settings, get_settings
from smart_security.core.exceptions import ModelLoadError
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)


class YoloDetector:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError("ultralytics is not installed") from exc

        model_path = self.settings.yolo_model_path
        try:
            logger.info("Loading YOLOv8 model from %s", model_path)
            model = YOLO(str(model_path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelLoadError(f"YOLO model loading failed: {model_path}") from exc
        logger.info("YOLO model loaded: %s", model_path)
        return model

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        results = self.model(
            frame,
            verbose=False,
            conf=self.settings.detection.confidence_threshold,
            iou=self.settings.detection.iou_threshold,
        )
        detections: list[list[float]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(box.conf[0].cpu().numpy()),
                        float(box.cls[0].cpu().numpy()),
                    ]
                )
        return detections


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_strongsort(device: torch.device, settings: Settings | None = None):
    cfg = settings or get_settings()
    try:
        from boxmot import StrongSort
    except ImportError as exc:
        raise ModelLoadError(
            "boxmot is not importable; add the vendored boxmot/ to PYTHONPATH"
        ) from exc

    reid_path = cfg.reid_weights_path
    reid_weights: Path | None = reid_path if reid_path.exists() else None
    try:
        tracker = StrongSort(
            reid_weights=reid_weights,
            device=str(device),
            half=device.type == "cuda",
            min_hits=cfg.tracking.min_hits,
            max_age=cfg.tracking.max_age,
            max_iou_dist=cfg.tracking.max_iou_dist,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelLoadError("Tracker initialization failed") from exc
    logger.info("Tracker initialized on %s", device)
    return tracker
