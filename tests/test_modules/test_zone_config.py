# 模块：警戒区几何测试
# 作用：验证默认左半屏多边形与单应投影两种判定路径。

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_security.modules.zone.zone_config import ZoneConfig, default_polygon_norm

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def test_default_polygon_left_half(tmp_path: Path) -> None:
    config_path = tmp_path / "zone.json"
    zone = ZoneConfig(path=config_path)
    zone.expansion_px = 0.0
    zone.polygon_norm = default_polygon_norm()
    zone.save()
    assert zone.point_in_zone(100.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert not zone.point_in_zone(500.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_hot_reload_reads_new_polygon(tmp_path: Path) -> None:
    config_path = tmp_path / "zone.json"
    zone = ZoneConfig(path=config_path)
    zone.polygon_norm = default_polygon_norm()
    zone.expansion_px = 0.0
    zone.save()

    other = ZoneConfig(path=config_path)
    other.polygon_norm = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
    other.expansion_px = 0.0
    other.save()

    assert zone.load() is True
    assert not zone.point_in_zone(100.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert zone.point_in_zone(500.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_homography_identity_uses_ground_polygon(tmp_path: Path) -> None:
    config_path = tmp_path / "zone.json"
    zone = ZoneConfig(path=config_path)
    zone.expansion_px = 0.0
    zone.homography = np.eye(3, dtype=np.float64)
    zone.ground_polygon = np.array(
        [[0.0, 0.0], [320.0, 0.0], [320.0, 480.0], [0.0, 480.0]],
        dtype=np.float64,
    )
    zone.save()
    reloaded = ZoneConfig(path=config_path)
    assert reloaded.homography is not None
    assert reloaded.point_in_zone(10.0, 10.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert not reloaded.point_in_zone(400.0, 10.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_expansion_offsets_every_edge_uniformly(tmp_path: Path) -> None:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.polygon_norm = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    zone.expansion_px = 20.0
    # 右边 x=480，外扩 20px：470/495 一内一外；上边 y=120 同理。
    assert zone.point_in_zone(495.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert not zone.point_in_zone(505.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert zone.point_in_zone(320.0, 105.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert not zone.point_in_zone(320.0, 95.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_concave_polygon_notch_stays_outside(tmp_path: Path) -> None:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    # U 形：中间上方有缺口，外扩只应让缺口边缘收窄而不是被填平。
    zone.polygon_norm = [
        [0.1, 0.1],
        [0.4, 0.1],
        [0.4, 0.6],
        [0.6, 0.6],
        [0.6, 0.1],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.1, 0.9],
    ]
    zone.expansion_px = 10.0
    assert not zone.point_in_zone(320.0, 120.0, FRAME_WIDTH, FRAME_HEIGHT)
    assert zone.point_in_zone(320.0, 400.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_mask_rebuilds_after_polygon_change(tmp_path: Path) -> None:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.expansion_px = 0.0
    zone.polygon_norm = default_polygon_norm()
    assert zone.point_in_zone(100.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)
    zone.polygon_norm = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
    assert not zone.point_in_zone(100.0, 240.0, FRAME_WIDTH, FRAME_HEIGHT)


def test_points_outside_frame_use_nearest_edge(tmp_path: Path) -> None:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.expansion_px = 0.0
    zone.polygon_norm = default_polygon_norm()
    assert zone.point_in_zone(100.0, 480.0, FRAME_WIDTH, FRAME_HEIGHT)


def _ground_zone(tmp_path: Path) -> ZoneConfig:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.expansion_px = 0.0
    zone.homography = np.eye(3, dtype=np.float64)
    # 地面区域只覆盖画面底部一条带：站在带内的人只有脚落在区域里。
    zone.ground_polygon = np.array(
        [[0.0, 400.0], [640.0, 400.0], [640.0, 480.0], [0.0, 480.0]],
        dtype=np.float64,
    )
    return zone


def test_ground_plane_coverage_samples_only_feet(tmp_path: Path) -> None:
    zone = _ground_zone(tmp_path)
    coverage = zone.coverage_ratio(
        100.0, 120.0, 160.0, 470.0, FRAME_WIDTH, FRAME_HEIGHT
    )
    assert coverage == 1.0


def test_ground_polygon_across_horizon_falls_back_to_projection(
    tmp_path: Path,
) -> None:
    zone = _ground_zone(tmp_path)
    # 第三行让 y>=240 的像素 w 变号：地面多边形跨地平线，不能整体映射回图像。
    zone.homography = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0 / 240.0, 1.0]],
        dtype=np.float64,
    )
    zone.ground_polygon = np.array(
        [[-1e4, -1e4], [1e4, -1e4], [1e4, 1e4], [-1e4, 1e4]], dtype=np.float64
    )
    assert zone.region_mask(FRAME_WIDTH, FRAME_HEIGHT) is None
    assert zone.point_in_zone(100.0, 100.0, FRAME_WIDTH, FRAME_HEIGHT)
