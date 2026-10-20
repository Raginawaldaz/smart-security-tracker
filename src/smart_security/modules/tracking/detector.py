# 模块：YOLO 检测与设备选择
# 作用：隔离模型加载失败点，把权重路径和设备选择从主循环中拆出。

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smart_security.core.config import Settings, get_settings
from smart_security.core.exceptions import ModelLoadError
from smart_security.utils.logger import get_logger

if TYPE_CHECKING:
    import torch

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
        """返回 [x1, y1, x2, y2, conf, cls] 行；只保留 detection.classes 中的类别。"""
        results = self.model(
            frame,
            verbose=False,
            conf=self.settings.detection.confidence_threshold,
            iou=self.settings.detection.iou_threshold,
            classes=self.settings.detection.classes or None,
        )
        detections: list[list[float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            rows = np.column_stack(
                (
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                )
            )
            detections.extend(rows.astype(np.float64).tolist())
        return detections


def select_device() -> torch.device:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
