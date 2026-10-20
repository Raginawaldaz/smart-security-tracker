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
