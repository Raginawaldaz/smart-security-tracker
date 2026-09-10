"""
数据库初始化脚本
从项目根目录运行: python -m src.db.init_db
"""
import sqlite3
from pathlib import Path


def init_database():
    """初始化 SQLite 数据库，建表并写入示例数据"""
    # 数据目录始终放在项目根目录下的 data/
    project_root = Path(__file__).parent.parent.parent
    db_dir = project_root / 'data'
    db_dir.mkdir(exist_ok=True, parents=True)
    db_path = db_dir / 'database.db'

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alarm_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        zone_id         INTEGER NOT NULL,
        zone_name       TEXT    NOT NULL,
        track_id        INTEGER NOT NULL,
        alarm_type      TEXT    NOT NULL,
        linger_time     REAL,
        screenshot_path TEXT,
        created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        log_level   TEXT NOT NULL,
        message     TEXT NOT NULL,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    INSERT INTO alarm_events
        (timestamp, zone_id, zone_name, track_id, alarm_type, linger_time)
    VALUES
        ('2025-10-22T10:00:00', 1, '入口警戒区', 101, 'intrusion', NULL),
        ('2025-10-22T11:30:00', 2, '核心区域',   102, 'lingering', 15.5),
        ('2025-10-22T14:45:00', 1, '入口警戒区', 103, 'intrusion', NULL)
    ''')

    conn.commit()
    conn.close()
    print(f"✅ 数据库初始化完成: {db_path}")


if __name__ == "__main__":
    init_database()
