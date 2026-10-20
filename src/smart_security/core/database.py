# 模块：SQLite 数据访问
# 作用：集中建表与报警/日志读写，避免 SQL 散落和连接泄漏。

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterable

from smart_security.core.config import get_settings
from smart_security.core.exceptions import DatabaseError
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

CREATE_ALARM_EVENTS_SQL = """
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
"""

CREATE_SYSTEM_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS system_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    log_level   TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

INSERT_ALARM_EVENT_SQL = """
INSERT INTO alarm_events
    (timestamp, zone_id, zone_name, track_id, alarm_type, linger_time, screenshot_path)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SYSTEM_LOG_SQL = """
INSERT INTO system_logs (timestamp, log_level, message)
VALUES (?, ?, ?)
"""

INSERT_SAMPLE_ALARM_SQL = """
INSERT INTO alarm_events
    (timestamp, zone_id, zone_name, track_id, alarm_type, linger_time)
VALUES (?, ?, ?, ?, ?, ?)
"""

SAMPLE_ALARM_ROWS: tuple[tuple[Any, ...], ...] = (
    ("2025-10-22T10:00:00", 1, "入口警戒区", 101, "intrusion", None),
    ("2025-10-22T11:30:00", 2, "核心区域", 102, "lingering", 15.5),
    ("2025-10-22T14:45:00", 1, "入口警戒区", 103, "intrusion", None),
)


class DatabaseManager:
    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to open database: {self.db_path}") from exc
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError("Database operation failed") from exc
        finally:
            conn.close()

    def init_database(self) -> None:
        with self.connection() as conn:
            conn.execute(CREATE_ALARM_EVENTS_SQL)
            conn.execute(CREATE_SYSTEM_LOGS_SQL)
        logger.info("Database initialized at %s", self.db_path)

    def seed_sample_data(self) -> None:
        """仅在显式 --seed 时写入示例数据，避免每次初始化污染真实记录。"""
        with self.connection() as conn:
            conn.executemany(INSERT_SAMPLE_ALARM_SQL, SAMPLE_ALARM_ROWS)
        logger.info("Inserted %s sample alarm rows", len(SAMPLE_ALARM_ROWS))

    def save_alarm_event(
        self, alarm_info: dict[str, Any], screenshot_path: str | None = None
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                INSERT_ALARM_EVENT_SQL,
                (
                    alarm_info["timestamp"],
                    alarm_info["zone_id"],
                    alarm_info["zone_name"],
                    alarm_info["track_id"],
                    alarm_info["alarm_type"],
                    alarm_info.get("linger_time"),
                    screenshot_path,
                ),
            )
            row_id = int(cursor.lastrowid or 0)
        logger.info(
            "Saved alarm event id=%s type=%s track_id=%s",
            row_id,
            alarm_info["alarm_type"],
            alarm_info["track_id"],
        )
        return row_id

    def get_alarm_events(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        alarm_type: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if alarm_type:
            clauses.append("alarm_type = ?")
            params.append(alarm_type)
        query = (
            "SELECT * FROM alarm_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY timestamp DESC"
        )
        with self.connection() as conn:
            rows: Iterable[sqlite3.Row] = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def add_system_log(self, log_level: str, message: str) -> None:
        with self.connection() as conn:
            conn.execute(
                INSERT_SYSTEM_LOG_SQL,
                (datetime.now().isoformat(), log_level, message),
            )


def init_database(db_path: Path | None = None, seed: bool = False) -> Path:
    manager = DatabaseManager(db_path=db_path)
    if seed:
        manager.seed_sample_data()
    return manager.db_path
