# 模块：实时监控主循环
# 作用：串联检测、跟踪、区内判定、逗留报警与绘制，是摄像头流程的唯一编排点。

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.core.exceptions import CameraError
from smart_security.modules.alarm.alert_store import AlertStore
from smart_security.modules.alarm.constants import LINGER_LABEL_MIN_SECONDS
from smart_security.modules.alarm.lingering import LingeringDetector
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
from smart_security.modules.tracking.detector import (
    YoloDetector,
    build_strongsort,
    select_device,
)
from smart_security.modules.tracking.fragment_merge import (
    merge_fragment_person_detections,
)
from smart_security.modules.tracking.id_dedup import TrackIdDeduplicator
from smart_security.modules.tracking.trajectory import TrajectoryStore
from smart_security.modules.zone.membership import ZoneMembership
from smart_security.modules.zone.zone_config import ZoneConfig
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

MIN_POLYGON_POINTS = 3


class TrackingPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = select_device()
        logger.info("Using device: %s", self.device)

        self.zone = ZoneConfig()
        self.membership = ZoneMembership(self.zone)
        self.trajectories = TrajectoryStore()
        self.dedup = TrackIdDeduplicator()
        self.detector = YoloDetector(self.settings)
        self.tracker = build_strongsort(self.device, self.settings)
        self.alert_store = AlertStore()
        self.lingering = LingeringDetector(self.alert_store, time.time)

        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.settings.screenshots_dir
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.start_time = time.time()

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
        current_time = time.time()
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

    def _save_preview_frame(self, frame: np.ndarray) -> None:
        try:
            cv2.imwrite(str(self.settings.last_frame_path), frame)
        except cv2.error as exc:
            logger.warning("Failed to write preview frame: %s", exc)

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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        self.membership.set_frame_size(frame_width, frame_height)

        if self.frame_count % self.settings.zone.reload_interval_frames == 0:
            if self.zone.load():
                logger.info("警戒区配置已更新")
            self._save_preview_frame(frame)

        self.draw_detection_zone(frame)
        detections = self.detector.detect(frame)
        detections = merge_fragment_person_detections(
            detections, frame_width, frame_height
        )

        if detections:
            try:
                tracked = self.tracker.update(np.array(detections), frame)
                tracked = self.dedup.apply(
                    tracked, frame_width, frame_height, self.frame_count
                )
                self.membership.update(tracked, self.frame_count)
                self.trajectories.update(tracked)
                self.lingering.check(tracked, frame, self.membership.is_in_zone)
                self._draw_tracks(frame, tracked)
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.exception("Tracking error: %s", exc)
        else:
            self.trajectories.update([])

        self.trajectories.draw(frame, self.membership.is_in_zone)
        self.draw_lingering_info(frame)
        return frame

    def run(self, camera_id: int | str | None = None) -> None:
        source = camera_id if camera_id is not None else self.settings.camera.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise CameraError(f"Cannot open camera: {source}")

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise CameraError("Cannot read camera frame")

        height, width = first_frame.shape[:2]
        self.membership.set_frame_size(width, height)
        self._save_preview_frame(first_frame)
        logger.info(
            "Zone polygon points: %s | ground-homography: %s",
            len(self.zone.polygon_norm),
            "on" if self.zone.homography is not None else "off",
        )
        logger.info(
            "Starting real-time monitoring — press 'q' to quit, 's' to screenshot"
        )

        prev_time = time.time()
        fps_buffer: list[float] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read error")
                    break

                current_time = time.time()
                delta = current_time - prev_time
                if delta > 0:
                    fps_buffer.append(1.0 / delta)
                    if len(fps_buffer) > FPS_BUFFER_SIZE:
                        fps_buffer.pop(0)
                prev_time = current_time
                avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

                frame = self.process_frame(frame)
                self._draw_hud(frame, avg_fps)
                cv2.imshow("Smart Security - Zone-Based Lingering Alert", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                if key == SCREENSHOT_KEY:
                    filename = f"manual_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(self.screenshots_dir / filename), frame)
                    logger.info("Screenshot saved: %s", filename)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            total_time = time.time() - self.start_time
            avg = self.frame_count / total_time if total_time > 0 else 0.0
            logger.info(
                "Final stats — frames: %s | avg FPS: %.1f", self.frame_count, avg
            )
            logger.info("System shutdown complete")
