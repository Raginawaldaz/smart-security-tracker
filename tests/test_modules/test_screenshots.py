# 模块：截图文件名解析测试
# 作用：仪表板依赖文件名中的 track_id 与时间戳，解析规则不能悄悄改变。

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from smart_security.modules.dashboard.screenshots import ScreenshotService


def test_parse_lingering_filename(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    (shots / "lingering_12_20260102_153045.jpg").write_bytes(b"fake")
    alerts = ScreenshotService(screenshots_dir=shots).list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 12
    assert alerts[0]["timestamp"] == datetime(2026, 1, 2, 15, 30, 45)


def test_unparsed_filename_uses_mtime(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    (shots / "manual_screenshot_1.jpg").write_bytes(b"fake")
    alerts = ScreenshotService(screenshots_dir=shots).list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 0
