import sqlite3
import json
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    def __init__(self):
        self.db_path = Path('data/database.db')
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建报警记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alarm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            zone_id INTEGER NOT NULL,
            zone_name TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            alarm_type TEXT NOT NULL,
            linger_time REAL,
            screenshot_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 创建系统日志表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()
        print("✅ 数据库初始化完成")

    def save_alarm_event(self, alarm_info, screenshot_path=None):
        """保存报警事件到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO alarm_events 
            (timestamp, zone_id, zone_name, track_id, alarm_type, linger_time, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alarm_info['timestamp'],
                alarm_info['zone_id'],
                alarm_info['zone_name'],
                alarm_info['track_id'],
                alarm_info['alarm_type'],
                alarm_info.get('linger_time'),
                screenshot_path
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ 保存报警事件失败: {e}")
            return False

    def get_alarm_events(self, start_time=None, end_time=None, alarm_type=None):
        """查询报警事件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM alarm_events WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if alarm_type:
                query += " AND alarm_type = ?"
                params.append(alarm_type)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            results = cursor.fetchall()

            # 转换为字典列表
            columns = [desc[0] for desc in cursor.description]
            events = [dict(zip(columns, row)) for row in results]

            conn.close()
            return events
        except Exception as e:
            print(f"❌ 查询报警事件失败: {e}")
            return []

    def add_system_log(self, log_level, message):
        """添加系统日志"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO system_logs (timestamp, log_level, message)
            VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), log_level, message))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ 添加系统日志失败: {e}")