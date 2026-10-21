# 模块：相机几何地面 3D 测试
# 作用：锁定高度/俯仰/视场角单应的方向，以及脚在区外、身体探进画面时不算进入。

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_security.modules.zone.camera import (
    image_to_ground_homography,
    project_pixels_to_ground,
)
from smart_security.modules.zone.membership import ZoneMembership
from smart_security.modules.zone.zone_config import ZoneConfig

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _camera_homography() -> np.ndarray:
    return image_to_ground_homography(FRAME_WIDTH, FRAME_HEIGHT, 3.0, 30.0, 70.0, 0.0)


def test_bottom_center_projects_in_front_of_camera() -> None:
    ground = project_pixels_to_ground(
        _camera_homography(), np.array([[FRAME_WIDTH / 2, FRAME_HEIGHT - 1.0]])
    )[0]
    assert np.isfinite(ground).all()
    assert ground[1] > 0
    assert abs(ground[0]) < 0.15


def test_higher_in_image_is_farther_on_ground() -> None:
    homography = _camera_homography()
    near = project_pixels_to_ground(
        homography, np.array([[FRAME_WIDTH / 2, FRAME_HEIGHT - 1.0]])
    )[0]
    farther = project_pixels_to_ground(
        homography, np.array([[FRAME_WIDTH / 2, FRAME_HEIGHT * 0.72]])
    )[0]
    assert farther[1] > near[1]


def test_enable_camera_ground_persists(tmp_path: Path) -> None:
    path = tmp_path / "zone.json"
    zone = ZoneConfig(path=path)
    zone.polygon_norm = [[0.25, 0.55], [0.75, 0.55], [0.95, 0.98], [0.05, 0.98]]
    zone.camera_height_m = 3.0
    zone.pitch_deg = 30.0
    zone.hfov_deg = 70.0
    assert zone.enable_camera_ground(FRAME_WIDTH, FRAME_HEIGHT)
    zone.save()

    reloaded = ZoneConfig(path=path)
    assert reloaded.ground_source == "camera"
    assert reloaded.ensure_ground_mapping(FRAME_WIDTH, FRAME_HEIGHT)
    assert reloaded.uses_ground_plane
    overlay = reloaded.overlay_polygon(FRAME_WIDTH, FRAME_HEIGHT)
    assert len(overlay) >= 3
    assert np.isfinite(overlay).all()


def _camera_zone(tmp_path: Path) -> ZoneConfig:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.expansion_px = 0.0
    # 画面底部中间一块，对应相机前方地面。
    zone.polygon_norm = [[0.2, 0.7], [0.8, 0.7], [0.95, 0.98], [0.05, 0.98]]
    zone.camera_height_m = 3.0
    zone.pitch_deg = 30.0
    zone.hfov_deg = 70.0
    assert zone.enable_camera_ground(FRAME_WIDTH, FRAME_HEIGHT)
    zone.save()
    return zone


def test_foot_inside_ground_zone_is_in(tmp_path: Path) -> None:
    membership = ZoneMembership(_camera_zone(tmp_path))
    membership.set_frame_size(FRAME_WIDTH, FRAME_HEIGHT)
    # 框底落在画面底部中央。
    membership.update([[250.0, 200.0, 390.0, 460.0, 1.0, 0.9, 0.0]], frame_count=1)
    assert membership.is_in_zone(1) is True


def test_body_overhang_without_feet_is_out(tmp_path: Path) -> None:
    membership = ZoneMembership(_camera_zone(tmp_path))
    membership.set_frame_size(FRAME_WIDTH, FRAME_HEIGHT)
    # 人站在画面右侧，脚在地面区外；上半身仍可能挡住绿色叠加。
    membership.update([[420.0, 40.0, 620.0, 250.0, 2.0, 0.9, 0.0]], frame_count=1)
    assert membership.is_in_zone(2) is False


def test_disable_ground_3d_returns_to_image_polygon(tmp_path: Path) -> None:
    zone = _camera_zone(tmp_path)
    zone.disable_ground_3d()
    zone.polygon_norm = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    zone.save()
    assert not zone.uses_ground_plane
    assert zone.point_in_zone(100.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert not zone.point_in_zone(500.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
