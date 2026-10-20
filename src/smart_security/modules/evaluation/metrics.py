# 模块：报警与 ID 稳定性指标
# 作用：把标注的逗留事件与系统报警对齐，计算准确率、召回率、延迟和每个事件内的 ID 数。

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True)
class LingerEvent:
    """一个人在警戒区内的一段连续停留（视频秒）。同时在区内的多人应分别标注。"""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class AlarmHit:
    time: float
    track_id: int


@dataclass
class AlarmMetrics:
    expected_events: int
    alarms: int
    detected_events: int
    true_alarms: int
    false_alarms: int
    latencies: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.true_alarms / self.alarms if self.alarms else None

    @property
    def recall(self) -> float | None:
        return (
            self.detected_events / self.expected_events
            if self.expected_events
            else None
        )

    @property
    def mean_latency(self) -> float | None:
        return sum(self.latencies) / len(self.latencies) if self.latencies else None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "expected_events": self.expected_events,
            "alarms": self.alarms,
            "detected_events": self.detected_events,
            "true_alarms": self.true_alarms,
            "false_alarms": self.false_alarms,
            "precision": self.precision,
            "recall": self.recall,
            "mean_latency_s": self.mean_latency,
        }


def load_events(path: Path) -> list[LingerEvent]:
    """标注格式：{"events": [{"start": 3.5, "end": 28.0}, ...]}，时间为视频秒。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    events = [
        LingerEvent(start=float(item["start"]), end=float(item["end"]))
        for item in data.get("events", [])
    ]
    return sorted(events, key=lambda event: event.start)


def score_alarms(
    events: Sequence[LingerEvent],
    alarms: Iterable[AlarmHit],
    lingering_threshold: float,
    tolerance: float = DEFAULT_TOLERANCE_SECONDS,
) -> AlarmMetrics:
    """
    时长达到阈值的事件应当报警：落在 [start+阈值-容差, end+容差] 内的报警算正确，
    同一事件的重复报警（周期性再报）也算正确但只计一次检出；其余报警都是误报。
    """
    expected = [event for event in events if event.duration >= lingering_threshold]
    windows = [
        (
            event.start + lingering_threshold - tolerance,
            event.end + tolerance,
            event.start + lingering_threshold,
        )
        for event in expected
    ]
    alarm_list = sorted(alarms, key=lambda alarm: alarm.time)
    first_hit: dict[int, float] = {}
    true_alarms = 0
    for alarm in alarm_list:
        matched = False
        for index, (window_start, window_end, due) in enumerate(windows):
            if window_start <= alarm.time <= window_end:
                matched = True
                first_hit.setdefault(index, alarm.time - due)
        if matched:
            true_alarms += 1
    return AlarmMetrics(
        expected_events=len(expected),
        alarms=len(alarm_list),
        detected_events=len(first_hit),
        true_alarms=true_alarms,
        false_alarms=len(alarm_list) - true_alarms,
        latencies=[first_hit[index] for index in sorted(first_hit)],
    )


def zone_ids_per_event(
    zone_observations: Iterable[tuple[float, Iterable[int]]],
    events: Sequence[LingerEvent],
) -> list[int]:
    """每个事件期间出现过的区内 track ID 数；理想值为 1，多出的就是 ID 断裂/切换。"""
    seen: list[set[int]] = [set() for _ in events]
    for timestamp, track_ids in zone_observations:
        ids = list(track_ids)
        for index, event in enumerate(events):
            if event.start <= timestamp <= event.end:
                seen[index].update(ids)
    return [len(ids) for ids in seen]
