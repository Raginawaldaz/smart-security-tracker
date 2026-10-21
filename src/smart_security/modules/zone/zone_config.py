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
from smart_security.modules.zone.camera import (
    DEFAULT_HEIGHT_M,
    DEFAULT_HFOV_DEG,
    DEFAULT_PITCH_DEG,
    DEFAULT_ROLL_DEG,
    image_to_ground_homography,
    project_pixels_to_ground,
)
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EXPANSION_PX = 20.0
MIN_POLYGON_POINTS = 3
HOMOGRAPHY_SHAPE = (3, 3)
GROUND_W_EPSILON = 1e-9
DEFAULT_LOWER_BAND = 0.58
DEFAULT_GRID_X = 5
DEFAULT_GRID_Y = 4
GROUND_CONTACT_BAND = 0.08
GROUND_CONTACT_ROWS = 2
MASK_INSIDE = 255
MAX_PROJECTED_COORD = 1e6


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
        self.ground_source: str = "off"
        self.camera_height_m: float = DEFAULT_HEIGHT_M
        self.pitch_deg: float = DEFAULT_PITCH_DEG
        self.hfov_deg: float = DEFAULT_HFOV_DEG
        self.roll_deg: float = DEFAULT_ROLL_DEG
        self.expansion_px: float = float(settings.zone.expansion_px)
        self._mtime: float | None = None
        self._cached_mask: np.ndarray | None = None
        self._cached_mask_key: tuple | None = None
        self._mapped_size: tuple[int, int] | None = None
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
        self.ground_source = str(data.get("ground_source") or "off")
        if self.ground_source not in {"off", "camera", "points"}:
            self.ground_source = "camera" if self.homography is not None else "off"
        if self.homography is not None and self.ground_source == "off":
            self.ground_source = "points"
        self.camera_height_m = float(data.get("camera_height_m", DEFAULT_HEIGHT_M))
        self.pitch_deg = float(data.get("pitch_deg", DEFAULT_PITCH_DEG))
        self.hfov_deg = float(data.get("hfov_deg", DEFAULT_HFOV_DEG))
        self.roll_deg = float(data.get("roll_deg", DEFAULT_ROLL_DEG))
        self._mapped_size = None
        self._cached_mask = None
        self._cached_mask_key = None

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
                "ground_source": self.ground_source,
                "camera_height_m": float(self.camera_height_m),
                "pitch_deg": float(self.pitch_deg),
                "hfov_deg": float(self.hfov_deg),
                "roll_deg": float(self.roll_deg),
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

    @property
    def uses_ground_plane(self) -> bool:
        return self.homography is not None and self.ground_polygon is not None

    def overlay_polygon(self, width: int, height: int) -> np.ndarray:
        """监控叠加用多边形：地面 3D 时反投地面区域，否则用画面多边形。"""
        self.ensure_ground_mapping(width, height)
        if self.uses_ground_plane:
            projected = self._ground_polygon_in_image()
            if projected is not None:
                return projected.astype(np.float32)
        return self.image_polygon(width, height)

    def enable_camera_ground(self, width: int, height: int) -> bool:
        self.ground_source = "camera"
        return self.ensure_ground_mapping(width, height, force=True)

    def enable_point_ground(
        self, homography: np.ndarray, width: int, height: int
    ) -> bool:
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != HOMOGRAPHY_SHAPE:
            return False
        self.ground_source = "points"
        self.homography = matrix
        self._mapped_size = (int(width), int(height))
        return self._refresh_ground_polygon(width, height)

    def disable_ground_3d(self) -> None:
        self.ground_source = "off"
        self.homography = None
        self.ground_polygon = None
        self._mapped_size = None
        self._cached_mask = None
        self._cached_mask_key = None

    def ensure_ground_mapping(
        self, width: int, height: int, force: bool = False
    ) -> bool:
        """按当前分辨率刷新单应。相机几何模式会重算 H，使不同画幅仍用同一套米制区域。"""
        size = (int(width), int(height))
        if (
            not force
            and self._mapped_size == size
            and self.homography is not None
            and self.ground_polygon is not None
        ):
            return True
        if self.ground_source == "off":
            return False
        if self.ground_source == "camera":
            try:
                self.homography = image_to_ground_homography(
                    size[0],
                    size[1],
                    self.camera_height_m,
                    self.pitch_deg,
                    self.hfov_deg,
                    self.roll_deg,
                )
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                logger.warning("camera ground homography failed")
                return False
            self._mapped_size = size
            self._cached_mask = None
            self._cached_mask_key = None
            return self._refresh_ground_polygon(size[0], size[1])
        if self.homography is None:
            return False
        self._mapped_size = size
        if force or self.ground_polygon is None:
            self._cached_mask = None
            self._cached_mask_key = None
            return self._refresh_ground_polygon(size[0], size[1])
        return True

    def _refresh_ground_polygon(self, width: int, height: int) -> bool:
        if self.homography is None:
            self.ground_polygon = None
            return False
        pixels = self.image_polygon(width, height)
        ground = project_pixels_to_ground(self.homography, pixels)
        if not np.isfinite(ground).all() or len(ground) < MIN_POLYGON_POINTS:
            logger.warning(
                "zone polygon crosses the horizon; keep previous ground poly"
            )
            return self.ground_polygon is not None
        self.ground_polygon = ground
        return True

    def _mask_key(self, width: int, height: int) -> tuple:
        return (
            int(width),
            int(height),
            tuple(tuple(point) for point in self.polygon_norm),
            float(self.expansion_px),
            None if self.homography is None else self.homography.tobytes(),
            None if self.ground_polygon is None else self.ground_polygon.tobytes(),
        )

    def region_mask(self, width: int, height: int) -> np.ndarray | None:
        """
        当前分辨率下的警戒区掩码（非零即区内），配置或尺寸变化时才重建。
        地面多边形跨过地平线、无法映射回图像多边形时返回 None，改走逐点投影。
        """
        self.ensure_ground_mapping(width, height)
        key = self._mask_key(width, height)
        if key != self._cached_mask_key:
            self._cached_mask = self._build_mask(int(width), int(height))
            self._cached_mask_key = key
        return self._cached_mask

    def _build_mask(self, width: int, height: int) -> np.ndarray | None:
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.uses_ground_plane:
            polygon = self._ground_polygon_in_image()
            if polygon is None:
                return None
            cv2.fillPoly(mask, [polygon], MASK_INSIDE)
            return mask

        polygon = np.rint(self.image_polygon(width, height)).astype(np.int32)
        if len(polygon) < MIN_POLYGON_POINTS:
            return mask
        cv2.fillPoly(mask, [polygon], MASK_INSIDE)
        # 掩码膨胀等价于多边形各边等距外扩，凹多边形也不会变形。
        radius = int(round(self.expansion_px))
        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mask = cv2.dilate(mask, kernel)
        return mask

    def _ground_polygon_in_image(self) -> np.ndarray | None:
        """地面多边形经单应逆变换回图像；所有顶点在地平线同侧时映射保持为多边形。"""
        if self.homography is None or self.ground_polygon is None:
            return None
        try:
            inverse = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return None
        ground = np.asarray(self.ground_polygon, dtype=np.float64)
        homogeneous = np.column_stack((ground, np.ones(len(ground)))) @ inverse.T
        scale = homogeneous[:, 2]
        if np.any(np.abs(scale) < GROUND_W_EPSILON):
            return None
        if not (np.all(scale > 0) or np.all(scale < 0)):
            return None
        image = homogeneous[:, :2] / scale[:, None]
        if not np.isfinite(image).all() or np.abs(image).max() > MAX_PROJECTED_COORD:
            return None
        return np.rint(image).astype(np.int32)

    def _points_in_ground(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        pixels = np.column_stack((xs, ys, np.ones(len(xs))))
        projected = pixels @ np.asarray(self.homography, dtype=np.float64).T
        ground_poly = self.ground_polygon.astype(np.float32).reshape(-1, 1, 2)
        inside = np.zeros(len(xs), dtype=bool)
        for index, (gx, gy, gw) in enumerate(projected):
            if abs(gw) < GROUND_W_EPSILON:
                continue
            inside[index] = (
                cv2.pointPolygonTest(
                    ground_poly, (float(gx / gw), float(gy / gw)), False
                )
                >= 0
            )
        return inside

    def points_in_zone(
        self, xs: np.ndarray, ys: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """
        批量判断像素点是否在警戒区内。
        - 若配置了 homography + ground_polygon：按地面坐标判断（接近 3D），不做外扩。
        - 否则：图像空间多边形（含外扩）内判断。
        画面外的点按最近的边缘像素判断。
        """
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        mask = self.region_mask(width, height)
        if mask is None:
            return self._points_in_ground(xs, ys)
        cols = np.clip(np.rint(xs).astype(np.int64), 0, int(width) - 1)
        rows = np.clip(np.rint(ys).astype(np.int64), 0, int(height) - 1)
        return mask[rows, cols] > 0

    def point_in_zone(
        self, pixel_x: float, pixel_y: float, width: int, height: int
    ) -> bool:
        return bool(
            self.points_in_zone(
                np.array([pixel_x]), np.array([pixel_y]), width, height
            )[0]
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
        地面单应只对接触地面的点成立，该模式下只采样框底附近的窄带。
        """
        box_height = max(float(y2) - float(y1), 1.0)
        if self.uses_ground_plane:
            top = float(y2) - GROUND_CONTACT_BAND * box_height
            rows = GROUND_CONTACT_ROWS
        else:
            top = float(y1) + (1.0 - lower_band) * box_height
            rows = grid_y
        xs = np.linspace(float(x1) + 1, float(x2) - 1, grid_x)
        ys = np.linspace(top + 1, float(y2) - 1, rows)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        inside = self.points_in_zone(grid_xs, grid_ys, width, height)
        return float(inside.mean()) if inside.size else 0.0
