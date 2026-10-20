# 模块：针孔相机地面单应
# 作用：用安装高度、俯仰角和水平视场角，把图像像素与地面米制坐标互转。
# 世界系：X 右、Y 前方、Z 上；相机在 (0, 0, height)，低头为正俯仰。

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_HEIGHT_M = 3.0
DEFAULT_PITCH_DEG = 30.0
DEFAULT_HFOV_DEG = 70.0
DEFAULT_ROLL_DEG = 0.0
MIN_CAMERA_HEIGHT_M = 0.3
MAX_CAMERA_HEIGHT_M = 30.0
MIN_PITCH_DEG = 0.0
MAX_PITCH_DEG = 85.0
MIN_HFOV_DEG = 20.0
MAX_HFOV_DEG = 120.0
MIN_ROLL_DEG = -30.0
MAX_ROLL_DEG = 30.0
GROUND_W_EPSILON = 1e-9
MAX_GROUND_RANGE_M = 200.0


def rotation_world_to_camera(pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """世界(X右,Y前,Z上) → 相机(x右,y下,z前)，先低头再滚转。"""
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))
    world_to_cv = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    pitch_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
        dtype=np.float64,
    )
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    roll_z = np.array(
        [[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return roll_z @ pitch_x @ world_to_cv


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    focal = (float(width) * 0.5) / np.tan(np.deg2rad(float(hfov_deg)) * 0.5)
    return np.array(
        [
            [focal, 0.0, float(width) * 0.5],
            [0.0, focal, float(height) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ground_to_image_homography(
    width: int,
    height: int,
    camera_height_m: float,
    pitch_deg: float,
    hfov_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """地面 [X, Y, 1]（米）→ 像素 [u, v, 1]。"""
    rotation = rotation_world_to_camera(pitch_deg, roll_deg)
    intrinsics = camera_intrinsics(width, height, hfov_deg)
    # P_cam = R @ (X, Y, -H) = X*r1 + Y*r2 - H*r3
    mapping = np.column_stack(
        (rotation[:, 0], rotation[:, 1], -float(camera_height_m) * rotation[:, 2])
    )
    return intrinsics @ mapping


def image_to_ground_homography(
    width: int,
    height: int,
    camera_height_m: float,
    pitch_deg: float,
    hfov_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """像素 [u, v, 1] → 地面 [X, Y, 1]（米）。"""
    return np.linalg.inv(
        ground_to_image_homography(
            width, height, camera_height_m, pitch_deg, hfov_deg, roll_deg
        )
    )


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    homogeneous = np.column_stack((pts, np.ones(len(pts)))) @ matrix.T
    scale = homogeneous[:, 2]
    valid = np.abs(scale) >= GROUND_W_EPSILON
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = homogeneous[valid, :2] / scale[valid, None]
    return out


def project_pixels_to_ground(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    ground = apply_homography(homography, points)
    for index, (east, north) in enumerate(ground):
        if not np.isfinite(east) or not np.isfinite(north):
            continue
        if north <= 0 or abs(east) > MAX_GROUND_RANGE_M or north > MAX_GROUND_RANGE_M:
            ground[index] = (np.nan, np.nan)
    return ground


def project_ground_to_pixels(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full((len(np.asarray(points).reshape(-1, 2)), 2), np.nan)
    return apply_homography(inverse, points)


def ground_grid_segments(
    homography: np.ndarray,
    width: int,
    height: int,
    spacing_m: float = 1.0,
    max_lines: int = 24,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """地面 1m 网格投到画面上的线段，供标定预览。"""
    sample_y = np.linspace(height * 0.35, height - 1, 12)
    sample_x = np.linspace(0, width - 1, 12)
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    ground = project_pixels_to_ground(
        homography, np.column_stack((grid_x.ravel(), grid_y.ravel()))
    )
    valid = ground[np.isfinite(ground).all(axis=1)]
    if len(valid) < 4:
        return []
    min_x, min_y = np.floor(valid.min(axis=0) / spacing_m) * spacing_m
    max_x, max_y = np.ceil(valid.max(axis=0) / spacing_m) * spacing_m
    xs = np.arange(min_x, max_x + spacing_m * 0.5, spacing_m)
    ys = np.arange(min_y, max_y + spacing_m * 0.5, spacing_m)
    if len(xs) > max_lines:
        xs = np.linspace(min_x, max_x, max_lines)
    if len(ys) > max_lines:
        ys = np.linspace(min_y, max_y, max_lines)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for east in xs:
        segments.extend(
            _clip_ground_line(homography, width, height, east, min_y, east, max_y)
        )
    for north in ys:
        segments.extend(
            _clip_ground_line(homography, width, height, min_x, north, max_x, north)
        )
    return segments


def _clip_ground_line(
    homography: np.ndarray,
    width: int,
    height: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    samples: int = 8,
) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    t = np.linspace(0.0, 1.0, samples)
    ground = np.column_stack((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pixels = project_ground_to_pixels(homography, ground)
    for start, end in zip(pixels[:-1], pixels[1:], strict=True):
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        if not _segment_hits_frame(start, end, width, height):
            continue
        yield (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def _segment_hits_frame(
    start: np.ndarray, end: np.ndarray, width: int, height: int
) -> bool:
    for point in (start, end):
        if 0 <= point[0] < width and 0 <= point[1] < height:
            return True
    # 两端都在画面外时仍可能穿过画面，用包围盒粗判即可。
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))
    return max_x >= 0 and min_x < width and max_y >= 0 and min_y < height
