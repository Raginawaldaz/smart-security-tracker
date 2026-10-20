# 模块：实时监控主循环
# 作用：串联检测、跟踪、区内判定、逗留报警与绘制，是摄像头流程的唯一编排点。

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.core.exceptions import CameraError
from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.constants import LINGER_LABEL_MIN_SECONDS
from smart_security.modules.alarm.lingering import LingeringDetector
from smart_security.modules.alarm.schemas import AlertRecord
from smart_security.modules.tracking.capture import LatestFrameCapture
from smart_security.modules.tracking.clock import (
    FrameClock,
    VideoTimeline,
    is_file_source,
)
from smart_security.modules.tracking.constants import (
    FPS_BUFFER_SIZE,
    HUD_GREEN,
    HUD_RED,
    HUD_WHITE,
    IN_ZONE_BOX_COLOR,
    LINGERING_ORANGE,
    LINGERING_RED,
    QUIT_KEY,
    SCREENSHOT_KEY,
    TRACK_ROW_MIN_LEN,
    ZONE_FILL_COLOR,
    ZONE_OVERLAY_ALPHA,
    ZONE_OVERLAY_BASE,
    get_class_color,
)
from smart_security.modules.tracking.detector import YoloDetector, select_device
from smart_security.modules.tracking.fragment_merge import (
    merge_fragment_person_detections,
)
from smart_security.modules.tracking.id_dedup import TrackIdDeduplicator
from smart_security.modules.tracking.relink import TrackRelinker
from smart_security.modules.tracking.tracker_factory import (
    build_appearance_encoder,
    build_tracker,
)
from smart_security.modules.tracking.trajectory import TrajectoryStore
from smart_security.modules.zone.membership import ZoneMembership
from smart_security.modules.zone.zone_config import ZoneConfig
from smart_security.utils.background import BackgroundWriter
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

MIN_POLYGON_POINTS = 3
DETECTION_COLUMNS = 6
DET_INDEX_COLUMN = 7


class TrackingPipeline:
    def __init__(
        self,
        tracker_name: str | None = None,
        alert_store: AlertStore | None = None,
        zone_path: Path | None = None,
        save_preview: bool = True,
    ) -> None:
        self.settings = get_settings()
        self.device = select_device()
        logger.info("Using device: %s", self.device)

        self.clock = FrameClock()
        self.writer = BackgroundWriter()
        self.save_preview = save_preview
        self.zone = ZoneConfig(path=zone_path)
        self.membership = ZoneMembership(self.zone)
        self.trajectories = TrajectoryStore()
        self.relinker = TrackRelinker()
        self.dedup = TrackIdDeduplicator()
        self.detector = YoloDetector(self.settings)
        self.tracker_name = tracker_name or self.settings.tracking.tracker
        self.tracker = build_tracker(self.tracker_name, self.device, self.settings)
        self.encoder = build_appearance_encoder(
            self.tracker, self.device, self.settings
        )
        self.alert_store = alert_store or AlertStore(writer=self.writer)
        self.lingering = LingeringDetector(self.alert_store, self.clock)

        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.settings.screenshots_dir
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.start_time = time.time()
        self.last_tracked: list[list[float]] = []
        self.last_alerts: list[AlertRecord] = []
        self.last_timings: dict[str, float] = {}

    def close(self) -> None:
        self.writer.close()

    def draw_detection_zone(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        polygon = self.zone.image_polygon(width, height).astype(np.int32)
        if len(polygon) < MIN_POLYGON_POINTS:
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], ZONE_FILL_COLOR)
        cv2.addWeighted(overlay, ZONE_OVERLAY_ALPHA, frame, ZONE_OVERLAY_BASE, 0, frame)
        cv2.polylines(frame, [polygon], True, ZONE_FILL_COLOR, 3, cv2.LINE_AA)
        text_x, text_y = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        tag = (
            "DETECTION ZONE (3D)"
            if self.zone.homography is not None
            else "DETECTION ZONE"
        )
        cv2.putText(
            frame,
            tag,
            (text_x + 5, text_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ZONE_FILL_COLOR,
            2,
        )
        cv2.putText(
            frame,
            f"Lingering >{int(self.lingering.lingering_threshold)}s triggers alert",
            (text_x + 5, text_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ZONE_FILL_COLOR,
            1,
        )

    def draw_lingering_info(self, frame: np.ndarray) -> None:
        current_time = self.clock()
        for track_id, info in self.lingering.lingering_targets.items():
            linger_time = current_time - info["enter_time"]
            if linger_time <= LINGER_LABEL_MIN_SECONDS:
                continue
            last_point = self.trajectories.last_point(int(track_id))
            if last_point is None:
                continue
            last_x, last_y = last_point
            text = f"{linger_time:.1f}s"
            text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (int(last_x) - 5, int(last_y) - text_h - 30),
                (int(last_x) + text_w + 5, int(last_y) - 25),
                (0, 0, 0),
                -1,
            )
            color = (
                LINGERING_ORANGE
                if linger_time < self.lingering.lingering_threshold
                else LINGERING_RED
            )
            cv2.putText(
                frame,
                text,
                (int(last_x), int(last_y) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if info["alert_count"] > 0:
                alert_text = f"Alerts: {info['alert_count']}"
                alert_w, alert_h = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )[0]
                cv2.rectangle(
                    frame,
                    (int(last_x) - 5, int(last_y) - alert_h - 60),
                    (int(last_x) + alert_w + 5, int(last_y) - 55),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    alert_text,
                    (int(last_x), int(last_y) - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    LINGERING_RED,
                    2,
                )

    def _draw_hud(self, frame: np.ndarray, avg_fps: float) -> None:
        elapsed = time.time() - self.start_time
        active = len(self.lingering.lingering_targets)
        total_alerts = sum(
            info["alert_count"] for info in self.lingering.lingering_targets.values()
        )
        lines: list[tuple[int, str, tuple[int, int, int]]] = [
            (30, "Smart Security - Zone-Based Lingering Detection", HUD_GREEN),
            (50, f"FPS: {avg_fps:.1f} | Frame: {self.frame_count}", HUD_WHITE),
            (70, f"Targets: {active} | Alerts: {total_alerts}", HUD_RED),
            (
                90,
                f"Run time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                HUD_WHITE,
            ),
        ]
        for y_pos, text, color in lines:
            cv2.putText(
                frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    @staticmethod
    def _write_image(path: Path, frame: np.ndarray) -> None:
        try:
            if not cv2.imwrite(str(path), frame):
                logger.warning("Failed to write image: %s", path)
        except cv2.error as exc:
            logger.warning("Failed to write image %s: %s", path, exc)

    def _save_preview_frame(self, frame: np.ndarray) -> None:
        self.writer.submit(
            self._write_image, self.settings.last_frame_path, frame.copy()
        )

    def _draw_tracks(self, frame: np.ndarray, tracked: list[Any]) -> None:
        for obj in tracked:
            if len(obj) < TRACK_ROW_MIN_LEN:
                continue
            x1, y1, x2, y2, track_id, conf, cls = obj[:7]
            in_zone = self.membership.is_in_zone(int(track_id))
            color = IN_ZONE_BOX_COLOR if in_zone else get_class_color(int(cls))
            thickness = 3 if in_zone else 2
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
            )
            label = f"ID:{int(track_id)} {conf:.2f}"
            if in_zone:
                label += " [IN ZONE]"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def _features_by_track(
        self, rows: list[list[float]], embeddings: np.ndarray | None
    ) -> dict[int, np.ndarray]:
        if embeddings is None:
            return {}
        features: dict[int, np.ndarray] = {}
        for row in rows:
            if len(row) <= DET_INDEX_COLUMN:
                continue
            index = int(row[DET_INDEX_COLUMN])
            if 0 <= index < len(embeddings):
                features[int(row[4])] = embeddings[index]
        return features

    def _track(
        self,
        detections: list[list[float]],
        frame: np.ndarray,
        now: float,
        timings: dict[str, float],
    ) -> list[list[float]]:
        """无检测的帧也要喂给跟踪器，卡尔曼预测和 max_age 计数才会推进。"""
        frame_height, frame_width = frame.shape[:2]
        det_array = np.asarray(detections, dtype=np.float64).reshape(
            -1, DETECTION_COLUMNS
        )
        try:
            started = time.perf_counter()
            embeddings = (
                self.encoder.encode(det_array[:, :4], frame)
                if self.encoder is not None
                else None
            )
            timings["reid"] = time.perf_counter() - started
            started = time.perf_counter()
            output = np.asarray(self.tracker.update(det_array, frame, embeddings))
            timings["track"] = time.perf_counter() - started
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Tracking error: %s", exc)
            return []

        rows = output.tolist() if output.ndim == 2 else []
        features = self._features_by_track(rows, embeddings)
        rows, features = self.relinker.apply(rows, now, features)
        return self.dedup.apply(
            rows, frame_width, frame_height, self.frame_count, features
        )

    def process_frame(
        self, frame: np.ndarray, timestamp: float | None = None
    ) -> np.ndarray:
        """timestamp 为视频秒；None 表示实时流，用墙钟。"""
        now = self.clock.tick(timestamp)
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        self.membership.set_frame_size(frame_width, frame_height)

        if self.frame_count % self.settings.zone.reload_interval_frames == 0:
            if self.zone.load():
                logger.info("警戒区配置已更新")
            if self.save_preview:
                self._save_preview_frame(frame)

        # 检测与 ReID 必须在画叠加层之前完成，否则区域着色和文字会污染模型输入。
        timings: dict[str, float] = {}
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        detections = merge_fragment_person_detections(
            detections, frame_width, frame_height
        )
        timings["detect"] = time.perf_counter() - started
        tracked = self._track(detections, frame, now, timings)

        started = time.perf_counter()
        self.membership.update(tracked, self.frame_count)
        self.trajectories.update(tracked)
        self.draw_detection_zone(frame)
        self.last_alerts = self.lingering.check(
            tracked, frame, self.membership.is_in_zone
        )
        self._draw_tracks(frame, tracked)
        self.trajectories.draw(frame, self.membership.is_in_zone)
        self.draw_lingering_info(frame)
        timings["logic"] = time.perf_counter() - started

        self.last_tracked = tracked
        self.last_timings = timings
        return frame

    def run(self, camera_id: int | str | None = None) -> None:
        source = camera_id if camera_id is not None else self.settings.camera.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise CameraError(f"Cannot open camera: {source}")

        timeline = (
            VideoTimeline(cap.get(cv2.CAP_PROP_FPS)) if is_file_source(source) else None
        )
        reader = (
            LatestFrameCapture(cap)
            if timeline is None and self.settings.camera.threaded_capture
            else cap
        )
        try:
            self._run_loop(cap, reader, timeline)
        finally:
            reader.release()
            cv2.destroyAllWindows()
            self.close()
            total_time = time.time() - self.start_time
            avg = self.frame_count / total_time if total_time > 0 else 0.0
            logger.info(
                "Final stats — frames: %s | avg FPS: %.1f", self.frame_count, avg
            )
            logger.info("System shutdown complete")

    def _run_loop(
        self,
        cap: cv2.VideoCapture,
        reader: cv2.VideoCapture | LatestFrameCapture,
        timeline: VideoTimeline | None,
    ) -> None:
        ret, first_frame = reader.read()
        if not ret or first_frame is None:
            raise CameraError("Cannot read camera frame")
        if timeline is not None:
            timeline.next(cap.get(cv2.CAP_PROP_POS_MSEC))

        height, width = first_frame.shape[:2]
        self.membership.set_frame_size(width, height)
        if self.save_preview:
            self._save_preview_frame(first_frame)
        logger.info(
            "Zone polygon points: %s | ground-homography: %s | time base: %s",
            len(self.zone.polygon_norm),
            "on" if self.zone.homography is not None else "off",
            "video timestamps" if timeline is not None else "wall clock",
        )
        logger.info(
            "Starting real-time monitoring — press 'q' to quit, 's' to screenshot"
        )

        prev_time = time.time()
        fps_buffer: list[float] = []

        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                logger.warning("Frame read error")
                break
            timestamp = (
                timeline.next(cap.get(cv2.CAP_PROP_POS_MSEC))
                if timeline is not None
                else None
            )

            current_time = time.time()
            delta = current_time - prev_time
            if delta > 0:
                fps_buffer.append(1.0 / delta)
                if len(fps_buffer) > FPS_BUFFER_SIZE:
                    fps_buffer.pop(0)
            prev_time = current_time
            avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

            frame = self.process_frame(frame, timestamp)
            self._draw_hud(frame, avg_fps)
            cv2.imshow("Smart Security - Zone-Based Lingering Alert", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == QUIT_KEY:
                break
            if key == SCREENSHOT_KEY:
                filename = f"manual_screenshot_{int(time.time())}.jpg"
                self.writer.submit(
                    self._write_image, self.screenshots_dir / filename, frame
                )
                logger.info("Screenshot queued: %s", filename)
