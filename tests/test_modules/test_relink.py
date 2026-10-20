# 模块：断链重关联测试
# 作用：确认遮挡后新 ID 能接回旧 ID，外观不符、间隔过长或旧 ID 被找回时不误接。

from __future__ import annotations

import numpy as np

from smart_security.core.config import RelinkSettings
from smart_security.modules.tracking.relink import TrackRelinker


def _row(track_id: int, x: float = 100.0) -> list[float]:
    return [x, 50.0, x + 60.0, 250.0, float(track_id), 0.9, 0.0, 0.0]


def _feature(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _relinker() -> TrackRelinker:
    return TrackRelinker(
        RelinkSettings(
            enabled=True,
            max_gap_seconds=5.0,
            min_similarity=0.65,
            max_dist_ratio=0.6,
            max_speed_ratio=1.0,
            feature_ema=0.8,
        )
    )


def test_new_id_after_occlusion_inherits_old_id() -> None:
    relinker = _relinker()
    person = _feature(1.0, 0.1, 0.0)
    relinker.apply([_row(1)], now=0.0, features={1: person})
    relinker.apply([], now=1.0)
    rows, features = relinker.apply(
        [_row(7, x=110.0)], now=2.0, features={7: _feature(0.95, 0.15, 0.0)}
    )
    assert int(rows[0][4]) == 1
    assert 1 in features
    assert relinker.relink_count == 1

    rows, _ = relinker.apply([_row(7, x=112.0)], now=2.1)
    assert int(rows[0][4]) == 1


def test_different_appearance_is_not_relinked() -> None:
    relinker = _relinker()
    relinker.apply([_row(1)], now=0.0, features={1: _feature(1.0, 0.0, 0.0)})
    rows, _ = relinker.apply([_row(7)], now=1.0, features={7: _feature(0.0, 1.0, 0.0)})
    assert int(rows[0][4]) == 7


def test_gap_longer_than_window_is_not_relinked() -> None:
    relinker = _relinker()
    person = _feature(1.0, 0.0, 0.0)
    relinker.apply([_row(1)], now=0.0, features={1: person})
    rows, _ = relinker.apply([_row(7)], now=6.0, features={7: person})
    assert int(rows[0][4]) == 7


def test_far_away_new_track_is_not_relinked() -> None:
    relinker = _relinker()
    person = _feature(1.0, 0.0, 0.0)
    relinker.apply([_row(1, x=0.0)], now=0.0, features={1: person})
    rows, _ = relinker.apply([_row(7, x=600.0)], now=0.5, features={7: person})
    assert int(rows[0][4]) == 7


def test_visible_track_is_never_a_relink_target() -> None:
    relinker = _relinker()
    person = _feature(1.0, 0.0, 0.0)
    relinker.apply([_row(1)], now=0.0, features={1: person})
    rows, _ = relinker.apply(
        [_row(1), _row(7, x=120.0)], now=0.1, features={1: person, 7: person}
    )
    assert sorted(int(row[4]) for row in rows) == [1, 7]


def test_original_id_reclaims_identity_when_tracker_recovers_it() -> None:
    relinker = _relinker()
    person = _feature(1.0, 0.0, 0.0)
    relinker.apply([_row(1)], now=0.0, features={1: person})
    rows, _ = relinker.apply([_row(7)], now=1.0, features={7: person})
    assert int(rows[0][4]) == 1
    rows, _ = relinker.apply(
        [_row(1), _row(7, x=300.0)], now=1.1, features={1: person, 7: person}
    )
    ids = {int(row[0]): int(row[4]) for row in rows}
    assert ids[100] == 1
    assert ids[300] == 7


def test_without_features_only_close_tracks_relink() -> None:
    relinker = _relinker()
    relinker.apply([_row(1)], now=0.0)
    rows, _ = relinker.apply([_row(7, x=105.0)], now=0.5)
    assert int(rows[0][4]) == 1


def test_non_person_rows_pass_through() -> None:
    relinker = _relinker()
    car = [1.0, 2.0, 3.0, 4.0, 9.0, 0.5, 2.0, 0.0]
    rows, _ = relinker.apply([car], now=0.0)
    assert rows == [car]


def test_disabled_relinker_keeps_raw_ids() -> None:
    relinker = TrackRelinker(RelinkSettings(enabled=False))
    person = _feature(1.0, 0.0, 0.0)
    relinker.apply([_row(1)], now=0.0, features={1: person})
    rows, _ = relinker.apply([_row(7)], now=1.0, features={7: person})
    assert int(rows[0][4]) == 7
