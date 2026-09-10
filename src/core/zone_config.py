"""
警戒区配置（支持「3D / 地面投影」判定）。

设计要点
--------
1. 区域用**多边形**描述（而非简单的轴对齐矩形），坐标统一用**归一化 0~1**，
   与分辨率无关，便于 Web 后台拖拽编辑后直接保存。
2. 可选 **地面单应矩阵 (homography)**：把图像像素点投影到「地面平面」坐标，
   再在地面坐标系里判断是否进入区域。这就是俯视/斜视相机下接近 3D 的做法
   —— 判定的是「人脚站在地面的哪个位置」，而不是「像素框压到哪」。
3. 文件热加载：检测到 configs/zone_config.json 的 mtime 变化即自动重载，
   因此 Web 后台改完区域，监控端无需重启即可生效。

JSON 结构
---------
{
  "polygon_norm":   [[x,y], ...],   # 图像空间多边形，归一化 0~1（必填）
  "homography":     null | [[...],[...],[...]],  # 3x3，图像像素 -> 地面坐标
  "ground_polygon": null | [[gx,gy], ...],       # 地面坐标系下的区域（配合 homography）
  "expansion_px":   20,             # 图像多边形外扩像素（防边界抖动）
  "updated_at":     "ISO8601"
}
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / 'configs' / 'zone_config.json'


def default_polygon_norm():
    """默认：画面左半区（与历史行为一致），归一化坐标。"""
    return [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


class ZoneConfig:
    def __init__(self, path: Path = _CONFIG_PATH):
        self.path = Path(path)
        self.polygon_norm = default_polygon_norm()
        self.homography = None          # np.ndarray (3,3) 或 None
        self.ground_polygon = None      # np.ndarray (N,2) 或 None
        self.expansion_px = 20.0
        self._mtime = None
        self.load(force=True)

    # ------------------------------------------------------------------ io

    def load(self, force: bool = False) -> bool:
        """读取配置。返回是否发生了重载。"""
        try:
            if not self.path.exists():
                if force:
                    self.save()  # 写出默认，方便 Web 端编辑
                return False
            mtime = self.path.stat().st_mtime
            if not force and mtime == self._mtime:
                return False
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply(data)
            self._mtime = mtime
            return True
        except Exception as e:  # 配置损坏时保底用默认，不影响监控
            print(f"⚠️ zone_config 读取失败，使用默认区域: {e}")
            return False

    def _apply(self, data: dict):
        poly = data.get('polygon_norm') or default_polygon_norm()
        self.polygon_norm = [[float(p[0]), float(p[1])] for p in poly if len(p) >= 2]
        if len(self.polygon_norm) < 3:
            self.polygon_norm = default_polygon_norm()

        h = data.get('homography')
        self.homography = np.asarray(h, dtype=np.float64) if h else None
        if self.homography is not None and self.homography.shape != (3, 3):
            self.homography = None

        gp = data.get('ground_polygon')
        self.ground_polygon = (np.asarray(gp, dtype=np.float64)
                               if gp and len(gp) >= 3 else None)

        self.expansion_px = float(data.get('expansion_px', 20.0))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'polygon_norm': [[float(x), float(y)] for x, y in self.polygon_norm],
            'homography': (self.homography.tolist()
                           if self.homography is not None else None),
            'ground_polygon': (self.ground_polygon.tolist()
                               if self.ground_polygon is not None else None),
            'expansion_px': float(self.expansion_px),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            self._mtime = self.path.stat().st_mtime
        except Exception:
            pass

    # ------------------------------------------------------------------ geometry

    def image_polygon(self, w: int, h: int) -> np.ndarray:
        """归一化多边形 → 当前分辨率下的像素多边形 (N,2) float。"""
        return np.array([[p[0] * w, p[1] * h] for p in self.polygon_norm],
                        dtype=np.float32)

    def _expanded_image_polygon(self, w: int, h: int) -> np.ndarray:
        """围绕质心向外扩 expansion_px，缓解边界抖动。"""
        poly = self.image_polygon(w, h)
        if self.expansion_px <= 0 or len(poly) < 3:
            return poly
        c = poly.mean(axis=0)
        out = []
        for p in poly:
            v = p - c
            n = float(np.hypot(v[0], v[1]))
            if n < 1e-6:
                out.append(p)
            else:
                out.append(p + v / n * self.expansion_px)
        return np.array(out, dtype=np.float32)

    def _to_ground(self, px: float, py: float):
        """图像像素 -> 地面坐标（需 homography）。返回 (gx, gy) 或 None。"""
        if self.homography is None:
            return None
        v = self.homography @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    def point_in_zone(self, px: float, py: float, w: int, h: int) -> bool:
        """
        判断像素点是否在警戒区内。
        - 若配置了 homography + ground_polygon：投影到地面坐标后判断（接近 3D）。
        - 否则：图像空间多边形（含外扩）内判断。
        """
        if self.homography is not None and self.ground_polygon is not None:
            g = self._to_ground(px, py)
            if g is None:
                return False
            gp = self.ground_polygon.astype(np.float32).reshape(-1, 1, 2)
            return cv2.pointPolygonTest(gp, (float(g[0]), float(g[1])), False) >= 0

        poly = self._expanded_image_polygon(w, h).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0

    def coverage_ratio(self, x1, y1, x2, y2, w, h,
                       lower_band: float = 0.58,
                       grid_x: int = 5, grid_y: int = 4) -> float:
        """
        采样 bbox **下半身带**的网格点，统计落在区域内的比例（0~1）。
        只有手臂伸进区时下半身带几乎不在区内 → 比例低；半身进入 → 比例显著升高。
        """
        ax1, ay1, ax2, ay2 = float(x1), float(y1), float(x2), float(y2)
        bh = max(ay2 - ay1, 1.0)
        top = ay1 + (1.0 - lower_band) * bh
        xs = np.linspace(ax1 + 1, ax2 - 1, grid_x)
        ys = np.linspace(top + 1, ay2 - 1, grid_y)
        total = 0
        inside = 0
        for yy in ys:
            for xx in xs:
                total += 1
                if self.point_in_zone(xx, yy, w, h):
                    inside += 1
        return inside / max(total, 1)
