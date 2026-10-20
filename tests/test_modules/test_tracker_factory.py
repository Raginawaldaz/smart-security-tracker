# 模块：跟踪器工厂测试
# 作用：确认未知名拒绝、各跟踪器构造参数正确，外观编码器优先复用跟踪器内置 ReID。

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from smart_security.core.config import Settings
from smart_security.core.exceptions import ModelLoadError
from smart_security.modules.tracking import tracker_factory as factory


class _FakeReid:
    def get_features(self, xyxy, img):
        return np.ones((len(xyxy), 3), dtype=np.float32)


class _FakeBoxmot:
    @staticmethod
    def StrongSort(**kwargs):
        return SimpleNamespace(kind="strongsort", kwargs=kwargs, model=_FakeReid())

    @staticmethod
    def BotSort(**kwargs):
        return SimpleNamespace(kind="botsort", kwargs=kwargs, model=_FakeReid())

    @staticmethod
    def DeepOcSort(**kwargs):
        return SimpleNamespace(kind="deepocsort", kwargs=kwargs, model=_FakeReid())

    @staticmethod
    def BoostTrack(**kwargs):
        return SimpleNamespace(
            kind="boosttrack", kwargs=kwargs, reid_model=_FakeReid()
        )


def test_unknown_tracker_raises() -> None:
    with pytest.raises(ModelLoadError, match="Unknown tracker"):
        factory.build_tracker("unknown", SimpleNamespace(type="cpu"), Settings())


def test_build_tracker_routes_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boxmot", _FakeBoxmot)
    settings = Settings()
    settings.tracking.max_age = 42
    settings.tracking.min_hits = 2
    settings.detection.confidence_threshold = 0.33
    device = SimpleNamespace(type="cpu")

    built = {
        name: factory.build_tracker(name, device, settings)
        for name in ("strongsort", "botsort", "deepocsort", "boosttrack")
    }
    assert built["strongsort"].kwargs["max_age"] == 42
    assert built["botsort"].kwargs["with_reid"] is True
    assert built["deepocsort"].kwargs["det_thresh"] == 0.33
    assert built["boosttrack"].kwargs["with_reid"] is True


def test_appearance_encoder_prefers_tracker_backend() -> None:
    tracker = SimpleNamespace(model=_FakeReid())
    encoder = factory.build_appearance_encoder(
        tracker, SimpleNamespace(type="cpu"), Settings()
    )
    assert encoder is not None
    features = encoder.encode(
        np.array([[0.0, 0.0, 10.0, 20.0]]), np.zeros((40, 40, 3), dtype=np.uint8)
    )
    assert features is not None
    assert features.shape == (1, 3)


def test_appearance_encoder_empty_boxes_returns_none() -> None:
    encoder = factory.AppearanceEncoder(_FakeReid())
    assert encoder.encode(np.zeros((0, 4)), np.zeros((8, 8, 3), dtype=np.uint8)) is None
