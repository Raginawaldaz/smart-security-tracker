# 模块：离线评测入口
# 作用：在同一段标注视频上跑一个或多个跟踪器，输出报警准确率/召回率、ID 断裂与耗时对比。
#
# 用法：
#   python scripts/evaluate.py --video eval/a.mp4 --labels eval/a.json --zone eval/a_zone.json
#   python scripts/evaluate.py --video eval/a.mp4 --labels eval/a.json \
#       --trackers strongsort botsort deepocsort boosttrack --output reports/a.json
#
# 标注 JSON：{"events": [{"start": 3.5, "end": 28.0}, ...]}，每条是一个人在警戒区内的
# 连续停留区间（视频秒）；同时在区内的多人分别标注。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from smart_security.core.config import get_settings  # noqa: E402
from smart_security.core.exceptions import SmartSecurityError  # noqa: E402
from smart_security.modules.evaluation.metrics import load_events  # noqa: E402
from smart_security.utils.logger import get_logger, setup_logging  # noqa: E402
from smart_security.utils.patches import apply_all  # noqa: E402

TABLE_COLUMNS = (
    ("tracker", "tracker"),
    ("fps", "processing_fps"),
    ("p95 ms", "latency_ms_p95"),
    ("person IDs", "distinct_person_ids"),
    ("zone IDs", "distinct_zone_ids"),
    ("extra IDs", "extra_ids_in_events"),
    ("relinks", "relinks"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("latency s", "mean_latency_s"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation on a video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--zone",
        type=Path,
        default=None,
        help="Zone config JSON for this video (defaults to zone.config_path)",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=None,
        help="Trackers to compare (defaults to tracking.tracker)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _flatten(report: dict) -> dict:
    flat = dict(report)
    flat.update(report.get("alarm_metrics", {}))
    return flat


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(reports: list[dict]) -> None:
    rows = [[header for header, _ in TABLE_COLUMNS]]
    for report in reports:
        flat = _flatten(report)
        rows.append([_format(flat.get(key)) for _, key in TABLE_COLUMNS])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row in rows:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    apply_all()
    logger = get_logger("smart_security.evaluate")

    from smart_security.modules.evaluation.runner import evaluate_video

    settings = get_settings()
    events = load_events(args.labels) if args.labels else None
    trackers = args.trackers or [settings.tracking.tracker]
    reports: list[dict] = []
    for tracker in trackers:
        try:
            result = evaluate_video(
                args.video,
                tracker_name=tracker,
                zone_path=args.zone,
                max_frames=args.max_frames,
            )
        except SmartSecurityError:
            logger.exception("Evaluation failed for tracker %s", tracker)
            return 1
        reports.append(result.summary(events, settings.alarm.lingering_threshold))

    print_table(reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, ensure_ascii=False, indent=2)
        logger.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
