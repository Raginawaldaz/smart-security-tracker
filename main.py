# 模块：项目执行入口
# 作用：统一配置日志、路径与子命令，避免从任意工作目录启动时找不到包。

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


def _bootstrap_path() -> None:
    for path in (str(SRC_DIR), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _pythonpath() -> str:
    parts = [str(SRC_DIR), str(PROJECT_ROOT)]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart security tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    track = subparsers.add_parser("track", help="Run camera tracking")
    track.add_argument(
        "--camera",
        default=None,
        help="Camera index or video path (overrides config.yaml)",
    )

    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    init_db = subparsers.add_parser("init-db", help="Initialize SQLite database")
    init_db.add_argument(
        "--seed",
        action="store_true",
        help="Insert sample alarm rows (opt-in only)",
    )
    return parser


def _parse_camera(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def cmd_track(camera: str | None) -> int:
    from smart_security.core.exceptions import SmartSecurityError
    from smart_security.modules.tracking.pipeline import TrackingPipeline
    from smart_security.utils.logger import get_logger, setup_logging
    from smart_security.utils.patches import apply_all

    setup_logging()
    apply_all()
    logger = get_logger("smart_security")
    logger.info("Starting tracking pipeline")
    try:
        TrackingPipeline().run(_parse_camera(camera))
    except SmartSecurityError:
        logger.exception("Tracking pipeline failed")
        return 1
    return 0


def cmd_dashboard() -> int:
    from smart_security.utils.logger import get_logger, setup_logging

    setup_logging()
    logger = get_logger("smart_security")
    app_path = SRC_DIR / "smart_security" / "modules" / "dashboard" / "app.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    logger.info("Launching Streamlit dashboard: %s", app_path)
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def cmd_init_db(seed: bool) -> int:
    from smart_security.core.database import init_database
    from smart_security.core.exceptions import DatabaseError
    from smart_security.utils.logger import get_logger, setup_logging

    setup_logging()
    logger = get_logger("smart_security")
    try:
        db_path = init_database(seed=seed)
    except DatabaseError:
        logger.exception("Database initialization failed")
        return 1
    logger.info("Database ready at %s", db_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    _bootstrap_path()
    args = build_parser().parse_args(argv)
    if args.command == "track":
        return cmd_track(args.camera)
    if args.command == "dashboard":
        return cmd_dashboard()
    if args.command == "init-db":
        return cmd_init_db(args.seed)
    return 1


if __name__ == "__main__":
    sys.exit(main())
