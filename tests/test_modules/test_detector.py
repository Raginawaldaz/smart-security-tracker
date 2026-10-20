# 模块：YOLO 检测器测试
# 作用：确认 classes 过滤传入模型，框解析走向量化路径，不依赖真实权重。

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smart_security.core.config import DetectionSettings, Settings
from smart_security.modules.tracking import detector as detector_module


class _FakeBoxes:
    def __init__(self, rows: np.ndarray) -> None:
        self.xyxy = _Tensor(rows[:, :4])
        self.conf = _Tensor(rows[:, 4])
        self.cls = _Tensor(rows[:, 5])

    def __len__(self) -> int:
        return len(self.xyxy.data)


class _Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.data


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.rows = np.array(
            [[1.0, 2.0, 3.0, 4.0, 0.9, 0.0], [5.0, 6.0, 7.0, 8.0, 0.8, 2.0]],
            dtype=np.float64,
        )

    def __call__(self, frame, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=_FakeBoxes(self.rows))]


def test_detect_passes_classes_and_vectorizes(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    settings.detection = DetectionSettings(classes=[0, 2], confidence_threshold=0.4)
    fake = _FakeModel()
    monkeypatch.setattr(
        detector_module.YoloDetector, "_load_model", lambda self: fake
    )
    detector = detector_module.YoloDetector(settings)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert fake.kwargs is not None
    assert fake.kwargs["classes"] == [0, 2]
    assert fake.kwargs["conf"] == 0.4
    assert detections == [
        [1.0, 2.0, 3.0, 4.0, 0.9, 0.0],
        [5.0, 6.0, 7.0, 8.0, 0.8, 2.0],
    ]


def test_detect_empty_boxes_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    fake = _FakeModel()
    fake.rows = np.zeros((0, 6), dtype=np.float32)
    monkeypatch.setattr(
        detector_module.YoloDetector, "_load_model", lambda self: fake
    )
    detector = detector_module.YoloDetector(settings)
    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
