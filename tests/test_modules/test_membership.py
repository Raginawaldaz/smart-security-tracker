# 模块：区内滞回判定测试
# 作用：确认完全在左侧警戒区的人判为在区内，完全在右侧的人判为区外。

from __future__ import annotations

from pathlib import Path

from smart_security.modules.zone.membership import ZoneMembership
from smart_security.modules.zone.zone_config import ZoneConfig, default_polygon_norm


def _zone(tmp_path: Path) -> ZoneConfig:
    zone = ZoneConfig(path=tmp_path / "zone.json")
    zone.polygon_norm = default_polygon_norm()
    zone.expansion_px = 0.0
    zone.homography = None
    zone.ground_polygon = None
    zone.save()
    return zone


def test_left_box_is_in_zone(tmp_path: Path) -> None:
    membership = ZoneMembership(_zone(tmp_path))
    membership.set_frame_size(640, 480)
    tracked = [[20.0, 40.0, 120.0, 400.0, 1.0, 0.9, 0.0]]
    membership.update(tracked, frame_count=1)
    assert membership.is_in_zone(1) is True


def test_right_box_is_out_of_zone(tmp_path: Path) -> None:
    membership = ZoneMembership(_zone(tmp_path))
    membership.set_frame_size(640, 480)
    tracked = [[400.0, 40.0, 520.0, 400.0, 2.0, 0.9, 0.0]]
    membership.update(tracked, frame_count=1)
    assert membership.is_in_zone(2) is False
