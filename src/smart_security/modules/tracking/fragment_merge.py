# 模块：人员检测框碎片合并
# 作用：合并「同一人被竖切成左右两框」的低 IoU 碎片，常规 NMS 压不掉这种情况。

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from smart_security.core.config import FragmentMergeSettings, get_settings
from smart_security.modules.tracking.constants import (
    DETECTION_ROW_MIN_LEN,
    PERSON_CLASS_ID,
)

DetectionRow = list[float]


def rect_intersection_area(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def should_merge_person_pair(
    box_a: Sequence[float],
    box_b: Sequence[float],
    frame_width: int,
    frame_height: int,
    params: FragmentMergeSettings | None = None,
) -> bool:
    """
    判断两个 person 框是否像「同一目标被竖切的两块」：竖向对齐度高且横向可衔接。
    frame_height 保留是为了与历史调用签名兼容，当前判定只用宽度定横向间隙。
    """
    _ = frame_height
    merge_params = params or get_settings().fragment_merge
    x1_a, y1_a, x2_a, y2_a = [float(value) for value in box_a[:4]]
    x1_b, y1_b, x2_b, y2_b = [float(value) for value in box_b[:4]]
    height_a = max(y2_a - y1_a, 1.0)
    height_b = max(y2_b - y1_b, 1.0)
    width_a = max(x2_a - x1_a, 1.0)
    width_b = max(x2_b - x1_b, 1.0)

    intersection_y = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
    vertical_overlap = intersection_y / min(height_a, height_b)
    if vertical_overlap < merge_params.vert_iou_min:
        return False

    intersection = rect_intersection_area(
        x1_a, y1_a, x2_a, y2_a, x1_b, y1_b, x2_b, y2_b
    )
    union = width_a * height_a + width_b * height_b - intersection + 1e-6
    iou = intersection / union
    if iou >= merge_params.iou_min:
        return True

    gap_x = max(0.0, max(x1_a, x1_b) - min(x2_a, x2_b))
    gap_threshold = max(
        merge_params.x_gap_min_px, merge_params.x_gap_frac * frame_width
    )
    if gap_x > gap_threshold:
        return False

    # 同一目标被竖切的两块框宽通常接近；成人+小孩并排时宽差大，降低误并。
    width_similarity = min(width_a, width_b) / max(width_a, width_b)
    if iou < merge_params.iou_min and width_similarity < merge_params.width_sim_min:
        return False

    union_x1, union_y1 = min(x1_a, x1_b), min(y1_a, y1_b)
    union_x2, union_y2 = max(x2_a, x2_b), max(y2_a, y2_b)
    union_w, union_h = max(union_x2 - union_x1, 1.0), max(union_y2 - union_y1, 1.0)
    union_aspect = union_w / union_h
    aspect_a = width_a / height_a
    aspect_b = width_b / height_b
    narrow_strip = min(aspect_a, aspect_b) < merge_params.narrow_strip_ar

    if union_aspect > merge_params.union_wh_max and not narrow_strip:
        return False
    return True


def merge_fragment_person_detections(
    detections: list[DetectionRow],
    frame_width: int,
    frame_height: int,
    params: FragmentMergeSettings | None = None,
) -> list[DetectionRow]:
    """对 person 类做碎片合并，其它类别原样保留。"""
    if not detections or len(detections) < 2:
        return detections

    det = np.asarray(detections, dtype=np.float64)
    if det.ndim != 2 or det.shape[1] < DETECTION_ROW_MIN_LEN:
        return detections

    class_column = det[:, 5].astype(np.int32)
    person_mask = class_column == PERSON_CLASS_ID
    if int(np.sum(person_mask)) < 2:
        return detections

    person_indices = np.flatnonzero(person_mask)
    other_indices = np.flatnonzero(~person_mask)
    person_count = len(person_indices)
    parent = np.arange(person_count, dtype=np.int32)

    def find(index: int) -> int:
        index = int(index)
        if int(parent[index]) != index:
            parent[index] = find(int(parent[index]))
        return int(parent[index])

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    merge_params = params or get_settings().fragment_merge
    for i in range(person_count):
        for j in range(i + 1, person_count):
            index_a = int(person_indices[i])
            index_b = int(person_indices[j])
            if should_merge_person_pair(
                det[index_a], det[index_b], frame_width, frame_height, merge_params
            ):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(person_count):
        root = int(find(i))
        clusters.setdefault(root, []).append(int(person_indices[i]))

    merged_rows: list[DetectionRow] = []
    for _root, indexes in clusters.items():
        rows = det[np.array(indexes, dtype=np.int32)]
        x1 = float(np.min(rows[:, 0]))
        y1 = float(np.min(rows[:, 1]))
        x2 = float(np.max(rows[:, 2]))
        y2 = float(np.max(rows[:, 3]))
        conf = float(np.max(rows[:, 4]))
        merged_rows.append([x1, y1, x2, y2, conf, float(PERSON_CLASS_ID)])

    return merged_rows + det[other_indices].tolist()
