# 模块：主循环编排测试
# 作用：用假检测器/跟踪器验证空帧也推进跟踪、模型输入不含叠加层、ReID 特征经 det_ind 送到重关联。

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.tracking import pipeline as pipeline_module

FRAME_SHAPE = (240, 320, 3)
PERSON_BOX = [20.0, 40.0, 80.0, 200.0]


class FakeDetector:
    def __init__(self, _settings=None) -> None:
        self.script: list[list[list[float]]] = []
        self.seen_frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        self.seen_frames.append(frame.copy())
        return self.script.pop(0) if self.script else []


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.script: list[np.ndarray] = []

    def update(self, dets, img, embs=None):
        self.calls.append((dets, embs))
        return self.script.pop(0) if self.script else np.array([])


class FakeEncoder:
    def encode(self, xyxy: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
        if xyxy.size == 0:
            return None
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(xyxy), 1))


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tracker = FakeTracker()
    monkeypatch.setattr(
        pipeline_module, "select_device", lambda: SimpleNamespace(type="cpu")
    )
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "build_tracker", lambda *_args: tracker)
    monkeypatch.setattr(
        pipeline_module, "build_appearance_encoder", lambda *_args: FakeEncoder()
    )
    store = AlertStore(
        screenshots_dir=tmp_path / "shots",
        alerts_file=tmp_path / "alerts.json",
        save_screenshots=False,
    )
    instance = pipeline_module.TrackingPipeline(
        alert_store=store, zone_path=tmp_path / "zone.json", save_preview=False
    )
    yield instance
    instance.close()


def _frame() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def _track_row(track_id: int, det_index: int = 0) -> np.ndarray:
    return np.array([[*PERSON_BOX, float(track_id), 0.9, 0.0, float(det_index)]])


def test_tracker_updated_on_frames_without_detections(pipeline) -> None:
    for step in range(3):
        pipeline.process_frame(_frame(), timestamp=step * 0.1)
    tracker = pipeline.tracker
    assert len(tracker.calls) == 3
    assert all(dets.shape == (0, 6) for dets, _embs in tracker.calls)


def test_detector_sees_frame_before_overlays(pipeline) -> None:
    pipeline.detector.script = [[[*PERSON_BOX, 0.9, 0.0]]]
    pipeline.tracker.script = [_track_row(1)]
    output = pipeline.process_frame(_frame(), timestamp=0.0)
    assert not pipeline.detector.seen_frames[0].any()
    assert output.any()


def test_lingering_target_cleared_when_scene_becomes_empty(pipeline) -> None:
    timestamp = 0.0
    for _ in range(5):
        pipeline.detector.script = [[[*PERSON_BOX, 0.9, 0.0]]]
        pipeline.tracker.script = [_track_row(1)]
        pipeline.process_frame(_frame(), timestamp=timestamp)
        timestamp += 0.1
    assert 1 in pipeline.lingering.lingering_targets

    pipeline.process_frame(_frame(), timestamp=timestamp + 10.0)
    assert 1 not in pipeline.lingering.lingering_targets


def test_reid_features_route_to_relinker(pipeline) -> None:
    timestamp = 0.0
    for _ in range(3):
        pipeline.detector.script = [[[*PERSON_BOX, 0.9, 0.0]]]
        pipeline.tracker.script = [_track_row(1)]
        pipeline.process_frame(_frame(), timestamp=timestamp)
        timestamp += 0.1
    pipeline.process_frame(_frame(), timestamp=timestamp + 1.0)

    pipeline.detector.script = [[[*PERSON_BOX, 0.9, 0.0]]]
    pipeline.tracker.script = [_track_row(7)]
    pipeline.process_frame(_frame(), timestamp=timestamp + 1.5)

    _dets, embs = pipeline.tracker.calls[-1]
    assert embs is not None and embs.shape == (1, 3)
    assert [int(row[4]) for row in pipeline.last_tracked] == [1]
    assert pipeline.relinker.relink_count == 1
