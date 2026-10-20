# 模块：离线评测执行
# 作用：无界面逐帧跑视频，记录报警、区内 ID 与各阶段耗时，产出可对比的报告。

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smart_security.core.exceptions import CameraError
from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.evaluation.metrics import (
    AlarmHit,
    LingerEvent,
    score_alarms,
    zone_ids_per_event,
)
from smart_security.modules.tracking.clock import VideoTimeline
from smart_security.modules.tracking.constants import (
    PERSON_CLASS_ID,
    TRACK_ROW_MIN_LEN,
)
from smart_security.modules.tracking.pipeline import TrackingPipeline
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

LATENCY_PERCENTILE = 95
MS_PER_SECOND = 1000.0


@dataclass
class EvaluationResult:
    video: str
    tracker: str
    frames: int
    video_seconds: float
    wall_seconds: float
    latencies_ms: list[float] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)
    alarms: list[AlarmHit] = field(default_factory=list)
    person_ids: set[int] = field(default_factory=set)
    zone_ids: set[int] = field(default_factory=set)
    zone_observations: list[tuple[float, list[int]]] = field(default_factory=list)
    relinks: int = 0
    dedup_merges: int = 0
    dedup_splits: int = 0

    def summary(
        self, events: list[LingerEvent] | None, lingering_threshold: float
    ) -> dict[str, Any]:
        latencies = np.asarray(self.latencies_ms, dtype=np.float64)
        report: dict[str, Any] = {
            "video": self.video,
            "tracker": self.tracker,
            "frames": self.frames,
            "video_seconds": round(self.video_seconds, 2),
            "processing_fps": (
                round(self.frames / self.wall_seconds, 2) if self.wall_seconds else None
            ),
            "latency_ms_mean": (
                round(float(latencies.mean()), 2) if latencies.size else None
            ),
            "latency_ms_p95": (
                round(float(np.percentile(latencies, LATENCY_PERCENTILE)), 2)
                if latencies.size
                else None
            ),
            "stage_ms_mean": {
                stage: round(total / max(self.frames, 1), 2)
                for stage, total in self.stage_ms.items()
            },
            "distinct_person_ids": len(self.person_ids),
            "distinct_zone_ids": len(self.zone_ids),
            "relinks": self.relinks,
            "dedup_merges": self.dedup_merges,
            "dedup_splits": self.dedup_splits,
            "alarms": [
                {"time": round(alarm.time, 2), "track_id": alarm.track_id}
                for alarm in self.alarms
            ],
        }
        if events is not None:
            metrics = score_alarms(events, self.alarms, lingering_threshold)
            ids_per_event = zone_ids_per_event(self.zone_observations, events)
            report["alarm_metrics"] = metrics.to_dict()
            report["zone_ids_per_event"] = ids_per_event
            report["extra_ids_in_events"] = sum(
                max(count - 1, 0) for count in ids_per_event
            )
        return report


def evaluate_video(
    video_path: Path,
    tracker_name: str | None = None,
    zone_path: Path | None = None,
    max_frames: int | None = None,
) -> EvaluationResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CameraError(f"Cannot open video: {video_path}")
    timeline = VideoTimeline(cap.get(cv2.CAP_PROP_FPS))

    with tempfile.TemporaryDirectory(prefix="smart_security_eval_") as scratch:
        scratch_dir = Path(scratch)
        store = AlertStore(
            screenshots_dir=scratch_dir / "shots",
            alerts_file=scratch_dir / "alerts.json",
            save_screenshots=False,
        )
        pipeline = TrackingPipeline(
            tracker_name=tracker_name,
            alert_store=store,
            zone_path=zone_path,
            save_preview=False,
        )
        result = EvaluationResult(
            video=str(video_path),
            tracker=pipeline.tracker_name,
            frames=0,
            video_seconds=0.0,
            wall_seconds=0.0,
        )
        wall_start = time.perf_counter()
        try:
            while max_frames is None or result.frames < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = timeline.next(cap.get(cv2.CAP_PROP_POS_MSEC))
                started = time.perf_counter()
                pipeline.process_frame(frame, timestamp)
                result.latencies_ms.append(
                    (time.perf_counter() - started) * MS_PER_SECOND
                )
                _record_frame(result, pipeline, timestamp)
        finally:
            cap.release()
            pipeline.close()
        result.wall_seconds = time.perf_counter() - wall_start
        result.relinks = pipeline.relinker.relink_count
        result.dedup_merges = pipeline.dedup.merge_count
        result.dedup_splits = pipeline.dedup.split_count
    logger.info(
        "Evaluated %s with %s: %s frames", video_path, result.tracker, result.frames
    )
    return result


def _record_frame(
    result: EvaluationResult, pipeline: TrackingPipeline, timestamp: float
) -> None:
    result.frames += 1
    result.video_seconds = timestamp
    for stage, seconds in pipeline.last_timings.items():
        result.stage_ms[stage] = (
            result.stage_ms.get(stage, 0.0) + seconds * MS_PER_SECOND
        )
    in_zone: list[int] = []
    for row in pipeline.last_tracked:
        if len(row) < TRACK_ROW_MIN_LEN or int(row[6]) != PERSON_CLASS_ID:
            continue
        track_id = int(row[4])
        result.person_ids.add(track_id)
        if pipeline.membership.is_in_zone(track_id):
            in_zone.append(track_id)
    result.zone_ids.update(in_zone)
    result.zone_observations.append((timestamp, in_zone))
    result.alarms.extend(
        AlarmHit(time=timestamp, track_id=int(alert.track_id))
        for alert in pipeline.last_alerts
    )
