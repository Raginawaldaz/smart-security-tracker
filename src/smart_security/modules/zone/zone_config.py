# 模块：警戒区配置
# 作用：以归一化多边形（可选地面单应）描述警戒区，并按文件 mtime 热加载，
# 这样 Web 后台改完区域后监控端不必重启。

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smart_security.core.config import get_settings
from smart_security.core.exceptions import ZoneConfigError
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EXPANSION_PX = 20.0
MIN_POLYGON_POINTS = 3
HOMOGRAPHY_SHAPE = (3, 3)
GROUND_W_EPSILON = 1e-9
CENTROID_NORM_EPSILON = 1e-6
DEFAULT_LOWER_BAND = 0.58
DEFAULT_GRID_X = 5
DEFAULT_GRID_Y = 4


def default_polygon_norm() -> list[list[float]]:
    """默认：画面左半区（与历史行为一致），归一化坐标。"""
    return [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


class ZoneConfig:
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = Path(path) if path is not None else settings.zone_config_path
        self.polygon_norm: list[list[float]] = default_polygon_norm()
        self.homography: np.ndarray | None = None
        self.ground_polygon: np.ndarray | None = None
        self.expansion_px: float = float(settings.zone.expansion_px)
        self._mtime: float | None = None
        self.load(force=True)

    def load(self, force: bool = False) -> bool:
        """读取配置。返回是否发生了重载。"""
        try:
            if not self.path.exists():
                if force:
                    self.save()
                return False
            mtime = self.path.stat().st_mtime
            if not force and mtime == self._mtime:
                return False
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._apply(data)
            self._mtime = mtime
            return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            logger.warning("zone_config 读取失败，使用默认区域: %s", exc)
            return False

    def _apply(self, data: dict) -> None:
        poly = data.get("polygon_norm") or default_polygon_norm()
        self.polygon_norm = [
            [float(point[0]), float(point[1])] for point in poly if len(point) >= 2
        ]
        if len(self.polygon_norm) < MIN_POLYGON_POINTS:
            self.polygon_norm = default_polygon_norm()

        homography = data.get("homography")
        self.homography = (
            np.asarray(homography, dtype=np.float64) if homography else None
        )
        if self.homography is not None and self.homography.shape != HOMOGRAPHY_SHAPE:
            self.homography = None

        ground_polygon = data.get("ground_polygon")
        self.ground_polygon = (
            np.asarray(ground_polygon, dtype=np.float64)
            if ground_polygon and len(ground_polygon) >= MIN_POLYGON_POINTS
            else None
        )
        self.expansion_px = float(data.get("expansion_px", DEFAULT_EXPANSION_PX))

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "polygon_norm": [[float(x), float(y)] for x, y in self.polygon_norm],
                "homography": (
                    self.homography.tolist() if self.homography is not None else None
                ),
                "ground_polygon": (
                    self.ground_polygon.tolist()
                    if self.ground_polygon is not None
                    else None
                ),
                "expansion_px": float(self.expansion_px),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self._mtime = self.path.stat().st_mtime
        except OSError as exc:
            raise ZoneConfigError(f"Failed to save zone config: {self.path}") from exc

    def image_polygon(self, width: int, height: int) -> np.ndarray:
        """归一化多边形 → 当前分辨率下的像素多边形 (N,2) float。"""
        return np.array(
            [[point[0] * width, point[1] * height] for point in self.polygon_norm],
            dtype=np.float32,
        )

    def _expanded_image_polygon(self, width: int, height: int) -> np.ndarray:
        """围绕质心向外扩 expansion_px，缓解边界抖动。"""
        polygon = self.image_polygon(width, height)
        if self.expansion_px <= 0 or len(polygon) < MIN_POLYGON_POINTS:
            return polygon
        centroid = polygon.mean(axis=0)
        expanded: list[np.ndarray] = []
        for point in polygon:
            vector = point - centroid
            norm = float(np.hypot(vector[0], vector[1]))
            if norm < CENTROID_NORM_EPSILON:
                expanded.append(point)
            else:
                expanded.append(point + vector / norm * self.expansion_px)
        return np.array(expanded, dtype=np.float32)

    def _to_ground(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        if self.homography is None:
            return None
        vector = self.homography @ np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
        if abs(vector[2]) < GROUND_W_EPSILON:
            return None
        return float(vector[0] / vector[2]), float(vector[1] / vector[2])

    def point_in_zone(
        self, pixel_x: float, pixel_y: float, width: int, height: int
    ) -> bool:
        """
        判断像素点是否在警戒区内。
        - 若配置了 homography + ground_polygon：投影到地面坐标后判断（接近 3D）。
        - 否则：图像空间多边形（含外扩）内判断。
        """
        if self.homography is not None and self.ground_polygon is not None:
            ground_point = self._to_ground(pixel_x, pixel_y)
            if ground_point is None:
                return False
            ground_poly = self.ground_polygon.astype(np.float32).reshape(-1, 1, 2)
            return (
                cv2.pointPolygonTest(
                    ground_poly, (float(ground_point[0]), float(ground_point[1])), False
                )
                >= 0
            )

        polygon = self._expanded_image_polygon(width, height).reshape(-1, 1, 2)
        return (
            cv2.pointPolygonTest(polygon, (float(pixel_x), float(pixel_y)), False) >= 0
        )

    def coverage_ratio(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
        lower_band: float = DEFAULT_LOWER_BAND,
        grid_x: int = DEFAULT_GRID_X,
        grid_y: int = DEFAULT_GRID_Y,
    ) -> float:
        """
        采样 bbox 下半身带的网格点，统计落在区域内的比例（0~1）。
        只有手臂伸进区时下半身带几乎不在区内 → 比例低；半身进入 → 比例显著升高。
        """
        box_height = max(float(y2) - float(y1), 1.0)
        top = float(y1) + (1.0 - lower_band) * box_height
        xs = np.linspace(float(x1) + 1, float(x2) - 1, grid_x)
        ys = np.linspace(top + 1, float(y2) - 1, grid_y)
        total = 0
        inside = 0
        for sample_y in ys:
            for sample_x in xs:
                total += 1
                if self.point_in_zone(sample_x, sample_y, width, height):
                    inside += 1
        return inside / max(total, 1)
