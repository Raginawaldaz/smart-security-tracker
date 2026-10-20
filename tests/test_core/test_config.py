# 模块：配置优先级测试
# 作用：锁定环境变量 > .env > config.yaml > 默认值，防止加载顺序被改坏。

from __future__ import annotations

from pathlib import Path

from smart_security.core.config import Settings


def test_defaults_when_files_missing(tmp_path: Path) -> None:
    settings = Settings.load(
        yaml_path=tmp_path / "missing.yaml",
        dotenv_path=tmp_path / "missing.env",
        environ={},
    )
    assert settings.alarm.lingering_threshold == 15.0
    assert settings.detection.confidence_threshold == 0.3


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("alarm:\n  lingering_threshold: 20\n", encoding="utf-8")
    settings = Settings.load(
        yaml_path=yaml_file,
        dotenv_path=tmp_path / "missing.env",
        environ={},
    )
    assert settings.alarm.lingering_threshold == 20.0


def test_dotenv_overrides_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("alarm:\n  lingering_threshold: 20\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("LINGERING_THRESHOLD=25\n", encoding="utf-8")
    settings = Settings.load(
        yaml_path=yaml_file,
        dotenv_path=env_file,
        environ={},
    )
    assert settings.alarm.lingering_threshold == 25.0


def test_environ_overrides_dotenv(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("alarm:\n  lingering_threshold: 20\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("LINGERING_THRESHOLD=25\n", encoding="utf-8")
    settings = Settings.load(
        yaml_path=yaml_file,
        dotenv_path=env_file,
        environ={"LINGERING_THRESHOLD": "40"},
    )
    assert settings.alarm.lingering_threshold == 40.0


def test_camera_source_keeps_video_path(tmp_path: Path) -> None:
    settings = Settings.load(
        yaml_path=tmp_path / "missing.yaml",
        dotenv_path=tmp_path / "missing.env",
        environ={"CAMERA_SOURCE": "clips/demo.mp4"},
    )
    assert settings.camera.source == "clips/demo.mp4"


def test_algorithm_defaults_match_roadmap(tmp_path: Path) -> None:
    settings = Settings.load(
        yaml_path=tmp_path / "missing.yaml",
        dotenv_path=tmp_path / "missing.env",
        environ={},
    )
    assert settings.tracking.tracker == "strongsort"
    assert settings.zone.in_seconds == 0.15
    assert settings.zone.out_seconds == 0.35
    assert settings.id_dedup.persist_frames == 8
    assert settings.relink.enabled is True
    assert settings.camera.threaded_capture is True


def test_tracker_env_override(tmp_path: Path) -> None:
    settings = Settings.load(
        yaml_path=tmp_path / "missing.yaml",
        dotenv_path=tmp_path / "missing.env",
        environ={"TRACKER_TYPE": "botsort"},
    )
    assert settings.tracking.tracker == "botsort"
