# 模块：统一配置加载
# 作用：按环境变量 > .env > config.yaml > 默认值的优先级集中暴露配置，
# 避免业务代码各自调用 os.getenv 或读取 YAML。

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values, load_dotenv

from smart_security.core.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YAML_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"

# 环境变量到嵌套字段的映射；只列出允许从环境覆盖的键，防止随意注入未知配置。
ENV_OVERRIDES: dict[str, str] = {
    "CAMERA_SOURCE": "camera.source",
    "CAMERA_WIDTH": "camera.width",
    "CAMERA_HEIGHT": "camera.height",
    "CAMERA_FPS": "camera.fps",
    "DETECTION_CONFIDENCE_THRESHOLD": "detection.confidence_threshold",
    "DETECTION_IOU_THRESHOLD": "detection.iou_threshold",
    "YOLO_MODEL_PATH": "detection.model",
    "TRACKER_TYPE": "tracking.tracker",
    "TRACKING_MAX_AGE": "tracking.max_age",
    "TRACKING_MIN_HITS": "tracking.min_hits",
    "REID_WEIGHTS_PATH": "tracking.reid_weights",
    "LINGERING_THRESHOLD": "alarm.lingering_threshold",
    "SAVE_SCREENSHOTS": "alarm.save_screenshots",
    "ALERT_SOUND": "alarm.alert_sound",
    "ZONE_CONFIG_PATH": "zone.config_path",
    "LOG_LEVEL": "system.log_level",
    "DB_PATH": "paths.database",
}


@dataclass
class CameraSettings:
    source: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    threaded_capture: bool = True


@dataclass
class DetectionSettings:
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.48
    classes: list[int] = field(default_factory=lambda: [0])
    model: str = "models/yolov8n.pt"


@dataclass
class TrackingSettings:
    tracker: str = "strongsort"
    max_age: int = 100
    min_hits: int = 3
    max_iou_dist: float = 0.7
    reid_weights: str = "weights/osnet_x0_25_msmt17.pt"


@dataclass
class AlarmSettings:
    lingering_threshold: float = 15.0
    undetected_timeout_seconds: float = 5.0
    save_screenshots: bool = True
    alert_sound: bool = True
    max_alerts_history: int = 100
    screenshot_quality: int = 85
    zones_path: str = "data/alarm_zones.json"


@dataclass
class ZoneSettings:
    config_path: str = "data/zone_config.json"
    expansion_px: float = 20.0
    in_seconds: float = 0.15
    out_seconds: float = 0.35
    lower_body_ratio: float = 0.58
    score_ema_alpha: float = 0.38
    schmitt_high: float = 0.52
    schmitt_low: float = 0.30
    membership_ttl_frames: int = 45
    reload_interval_frames: int = 15


@dataclass
class TrajectorySettings:
    ema_alpha: float = 0.40
    min_move_px: float = 2.0
    keep_missing: int = 30
    max_len: int = 60
    anchor_bottom_lift_ratio: float = 0.12
    anchor_bottom_lift_min_px: float = 10.0
    anchor_lift_max_ratio: float = 0.38


@dataclass
class FragmentMergeSettings:
    vert_iou_min: float = 0.64
    x_gap_frac: float = 0.032
    x_gap_min_px: float = 14.0
    iou_min: float = 0.10
    union_wh_max: float = 0.78
    width_sim_min: float = 0.42
    narrow_strip_ar: float = 0.42


@dataclass
class IdDedupSettings:
    persist_frames: int = 8
    alias_ttl: int = 150
    min_similarity: float = 0.5
    split_frames: int = 10


@dataclass
class RelinkSettings:
    enabled: bool = True
    max_gap_seconds: float = 5.0
    min_similarity: float = 0.65
    max_dist_ratio: float = 0.6
    max_speed_ratio: float = 1.0
    feature_ema: float = 0.8


@dataclass
class SystemSettings:
    log_level: str = "INFO"
    auto_start: bool = False
    theme: str = "dark"


@dataclass
class PathSettings:
    data_dir: str = "data"
    logs_dir: str = "logs"
    screenshots_dir: str = "data/screenshots"
    alerts_file: str = "data/lingering_alerts.json"
    last_frame: str = "data/last_frame.jpg"
    database: str = "data/database.db"


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    alarm: AlarmSettings = field(default_factory=AlarmSettings)
    zone: ZoneSettings = field(default_factory=ZoneSettings)
    trajectory: TrajectorySettings = field(default_factory=TrajectorySettings)
    fragment_merge: FragmentMergeSettings = field(default_factory=FragmentMergeSettings)
    id_dedup: IdDedupSettings = field(default_factory=IdDedupSettings)
    relink: RelinkSettings = field(default_factory=RelinkSettings)
    system: SystemSettings = field(default_factory=SystemSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    def resolve(self, relative: str) -> Path:
        """相对路径一律锚定到项目根，避免从不同工作目录启动时读错文件。"""
        path = Path(relative)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def logs_dir(self) -> Path:
        return self.resolve(self.paths.logs_dir)

    @property
    def screenshots_dir(self) -> Path:
        return self.resolve(self.paths.screenshots_dir)

    @property
    def alerts_file(self) -> Path:
        return self.resolve(self.paths.alerts_file)

    @property
    def last_frame_path(self) -> Path:
        return self.resolve(self.paths.last_frame)

    @property
    def database_path(self) -> Path:
        return self.resolve(self.paths.database)

    @property
    def zone_config_path(self) -> Path:
        return self.resolve(self.zone.config_path)

    @property
    def alarm_zones_path(self) -> Path:
        return self.resolve(self.alarm.zones_path)

    @property
    def yolo_model_path(self) -> Path:
        return self.resolve(self.detection.model)

    @property
    def reid_weights_path(self) -> Path:
        return self.resolve(self.tracking.reid_weights)

    @classmethod
    def load(
        cls,
        yaml_path: Path | None = None,
        dotenv_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        settings = cls()
        yaml_file = yaml_path if yaml_path is not None else DEFAULT_YAML_PATH
        env_file = dotenv_path if dotenv_path is not None else DEFAULT_DOTENV_PATH

        yaml_data = _load_yaml(yaml_file)
        if yaml_data:
            _overlay_dataclass(settings, yaml_data)

        env_mapping = _resolve_environ(env_file, environ)
        _apply_env_overrides(settings, env_mapping)
        return settings


def _resolve_environ(
    env_file: Path, environ: Mapping[str, str] | None
) -> Mapping[str, str]:
    """合并 .env 与进程环境：显式传入 environ 时不污染 os.environ，便于单测。"""
    file_values: dict[str, str] = {}
    if env_file.exists():
        file_values = {
            key: value
            for key, value in dotenv_values(env_file).items()
            if value is not None
        }

    if environ is None:
        # override=False：已存在的操作系统环境变量不被 .env 覆盖。
        if env_file.exists():
            load_dotenv(env_file, override=False)
        return os.environ

    merged = dict(file_values)
    merged.update(dict(environ))
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to read config yaml: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config yaml must be a mapping: {path}")
    return loaded


def _overlay_dataclass(obj: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _overlay_dataclass(current, value)
            continue
        setattr(obj, key, _cast_value(value, current))


def _apply_env_overrides(settings: Settings, environ: Mapping[str, str]) -> None:
    for env_key, dotted in ENV_OVERRIDES.items():
        raw = environ.get(env_key)
        if raw is None or raw == "":
            continue
        _set_dotted(settings, dotted, raw)


def _set_dotted(settings: Settings, dotted: str, raw: str) -> None:
    parts = dotted.split(".")
    target: Any = settings
    for part in parts[:-1]:
        target = getattr(target, part)
    field_name = parts[-1]
    current = getattr(target, field_name)
    setattr(target, field_name, _cast_env(raw, current, field_name))


def _cast_value(value: Any, template: Any) -> Any:
    if isinstance(template, bool):
        if isinstance(value, bool):
            return value
        return _as_bool(str(value))
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, list):
        return list(value)
    return value


def _cast_env(raw: str, template: Any, field_name: str) -> Any:
    # 摄像头源既可以是设备编号也可以是视频路径，数字才转 int。
    if field_name == "source":
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(template, bool):
        return _as_bool(raw)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if isinstance(template, list):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ConfigError(
                f"Environment override for {field_name} must be a JSON list"
            )
        return parsed
    return raw


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
