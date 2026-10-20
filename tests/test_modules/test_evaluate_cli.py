# 模块：评测入口测试
# 作用：确认 CLI 能汇总多个跟踪器报告，并写出 JSON；不启动真实模型。

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from smart_security.modules.evaluation.metrics import AlarmHit, LingerEvent
from smart_security.modules.evaluation.runner import EvaluationResult


def _load_evaluate_cli():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_cli"] = module
    spec.loader.exec_module(module)
    return module


def test_print_table_and_main_write_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    evaluate_cli = _load_evaluate_cli()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"events": [{"start": 0.0, "end": 30.0}]}), encoding="utf-8"
    )
    output = tmp_path / "report.json"
    calls: list[str] = []

    def fake_evaluate(video_path, tracker_name=None, zone_path=None, max_frames=None):
        calls.append(tracker_name or "default")
        return EvaluationResult(
            video=str(video_path),
            tracker=tracker_name or "strongsort",
            frames=10,
            video_seconds=0.5,
            wall_seconds=0.2,
            latencies_ms=[5.0],
            alarms=[AlarmHit(15.2, 1)],
            person_ids={1},
            zone_ids={1},
            zone_observations=[(1.0, [1]), (5.0, [1])],
        )

    monkeypatch.setattr(
        "smart_security.modules.evaluation.runner.evaluate_video", fake_evaluate
    )
    monkeypatch.setattr(evaluate_cli, "apply_all", lambda: None)
    monkeypatch.setattr(evaluate_cli, "setup_logging", lambda: None)

    code = evaluate_cli.main(
        [
            "--video",
            str(video),
            "--labels",
            str(labels),
            "--trackers",
            "strongsort",
            "botsort",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert calls == ["strongsort", "botsort"]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert [row["tracker"] for row in written] == ["strongsort", "botsort"]
    printed = capsys.readouterr().out
    assert "strongsort" in printed
    assert "botsort" in printed
    assert "precision" in printed

    summary = EvaluationResult(
        video="demo.mp4",
        tracker="strongsort",
        frames=30,
        video_seconds=1.0,
        wall_seconds=0.5,
        latencies_ms=[10.0],
        alarms=[AlarmHit(15.2, 1)],
        person_ids={1},
        zone_ids={1},
        zone_observations=[(1.0, [1])],
    ).summary([LingerEvent(0.0, 30.0)], lingering_threshold=15.0)
    assert summary["alarm_metrics"]["detected_events"] == 1
