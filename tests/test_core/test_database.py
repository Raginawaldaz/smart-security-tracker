# 模块：数据库读写测试
# 作用：验证建表、报警插入与条件查询，且 --seed 才写入示例行。

from __future__ import annotations

from pathlib import Path

from smart_security.core.database import DatabaseManager, init_database
from smart_security.modules.alarm.constants import AlarmType


def test_init_without_seed_is_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "database.db"
    init_database(db_path=db_path, seed=False)
    manager = DatabaseManager(db_path=db_path)
    assert manager.get_alarm_events() == []


def test_seed_inserts_sample_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "database.db"
    init_database(db_path=db_path, seed=True)
    manager = DatabaseManager(db_path=db_path)
    events = manager.get_alarm_events()
    assert len(events) == 3


def test_save_and_filter_alarm_event(tmp_path: Path) -> None:
    manager = DatabaseManager(db_path=tmp_path / "database.db")
    row_id = manager.save_alarm_event(
        {
            "timestamp": "2026-01-01T10:00:00",
            "zone_id": 1,
            "zone_name": "入口警戒区",
            "track_id": 7,
            "alarm_type": AlarmType.LINGERING.value,
            "linger_time": 16.2,
        },
        screenshot_path="data/screenshots/demo.jpg",
    )
    assert row_id > 0
    lingering = manager.get_alarm_events(alarm_type=AlarmType.LINGERING.value)
    assert len(lingering) == 1
    assert lingering[0]["track_id"] == 7
    assert manager.get_alarm_events(alarm_type=AlarmType.INTRUSION.value) == []
