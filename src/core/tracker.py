"""
智能安防核心追踪模块
运行方式（两种均可）:
  python src/core/tracker.py
  python -m src.core.tracker
"""
import cv2
import torch
from pathlib import Path
import sys
import numpy as np
import time
from datetime import datetime
import json

# 项目根目录（无论从哪里运行都能正确定位）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 确保项目根目录和 boxmot 都在搜索路径里
# 直接 `python src/core/tracker.py` 和 `python -m src.core.tracker` 都能跑
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / 'boxmot') not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / 'boxmot'))

# 在导入 YOLO 前应用补丁
from src.utils.patches import apply_all
apply_all()

from src.core.zone_config import ZoneConfig

print("🚀 Starting Smart Security System")

try:
    from ultralytics import YOLO
    from boxmot import StrongSort
    print("✅ All dependencies imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 数据目录统一放在项目根目录
_DATA_DIR = _PROJECT_ROOT / 'data'


class MultiObjectTracker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Lingering detection settings
        self.lingering_threshold = 15  # seconds
        self.lingering_targets = {}
        self.zone_enter_buffer = {}

        # 全画面人员轨迹 { track_id: [(cx, cy), ...] }
        self.person_tracks: dict = {}
        # 轨迹丢失计数：短时漏检不立即清理轨迹，减少断裂
        self.track_miss_counts: dict = {}
        # 轨迹平滑参数
        self.track_ema_alpha = 0.40       # 稍大一点，减少滞后
        self.track_min_move_px = 2.0      # 灵敏一点
        self.track_keep_missing = 30      # 增加到30帧，约1-2秒，防止频繁断开
        self.track_max_len = 60           # 轨迹长一些，视觉效果更好
        # 商业监控画面：轨迹不取 bbox 最底边 y2（常贴在画面下沿/被任务栏裁切），
        # 改为「足底略上方」的人体锚点，尾迹仍表示行走路径且始终在可视区内。
        self.track_anchor_bottom_lift_ratio = 0.12   # 相对框高向上抬的比例
        self.track_anchor_bottom_lift_min_px = 10.0 # 至少抬高像素，小框也可见

        # 警戒区：多边形 + 可选地面单应（3D 投影），由 configs/zone_config.json 热加载，
        # Web 后台拖拽编辑后无需重启即可生效。
        self.zone = ZoneConfig()
        self.frame_w = None
        self.frame_h = None

        # 区域进出缓冲：防止边界抖动
        self._ZONE_IN_FRAMES  = 4
        self._ZONE_OUT_FRAMES = 10
        self._zone_in_counter:  dict = {}   # {track_id: 连续在区域内的帧数}
        self._zone_out_counter: dict = {}   # {track_id: 连续不在区域内的帧数}

        # 俯视/高位相机：用「足底 + 下半身带 + 框重叠」综合判定是否在警戒区内，
        # 再经 EMA + Schmitt 滞回，避免身体跨边界时整框在绿/非绿之间来回跳（像两条轨迹）。
        self._zone_lower_body_ratio = 0.58   # 取 bbox 靠下 58% 作为躯干+腿（手臂主要在上方）
        self._zone_score_ema_alpha = 0.38
        self._zone_schmitt_high = 0.52       # 高于此 → 判为在区内
        self._zone_schmitt_low = 0.30        # 低于此 → 判为在区外；中间保持上一状态
        self._zone_score_ema: dict = {}      # {track_id: float}
        self._zone_hyst_in: dict = {}        # {track_id: bool} 滞回后的在区状态
        self._zone_last_seen_frame: dict = {}  # {track_id: int} 用于清理

        # Data directories
        self.data_dir = _DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        self.screenshots_dir = self.data_dir / 'screenshots'
        self.screenshots_dir.mkdir(exist_ok=True)
        self.alerts_file = self.data_dir / 'lingering_alerts.json'

        # Load YOLO model
        try:
            print("Loading YOLOv8 model...")
            model_path = _PROJECT_ROOT / 'models' / 'yolov8n.pt'
            self.yolo_model = YOLO(str(model_path))
            print(f"✅ YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"❌ YOLO model loading failed: {e}")
            sys.exit(1)

        # Initialize tracker
        try:
            reid_weights_path = _PROJECT_ROOT / 'weights' / 'osnet_x0_25_msmt17.pt'
            self.tracker = StrongSort(
                reid_weights=reid_weights_path if reid_weights_path.exists() else None,
                device=str(self.device),
                half=True if self.device.type == 'cuda' else False,
                min_hits=3,       # 降低一点，让新目标更快出现但仍有一定过滤
                max_age=100,      # 大幅增加到100帧，防止人走出一半被切断
                max_iou_distance=0.7, # 增加容忍度
            )
            print("✅ Tracker initialized successfully")
        except Exception as e:
            print(f"❌ Tracker initialization failed: {e}")
            sys.exit(1)

        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.class_colors = {
            'person':  (0, 165, 255),   # Orange
            'vehicle': (255, 0, 0),     # Blue
            'animal':  (0, 255, 0),     # Green
            'other':   (255, 255, 0),   # Yellow
        }

        self.frame_count = 0
        self.start_time = time.time()

        # 送入跟踪器前：合并「同一人被拆成左右两框」类碎片（IoU 常很低，内置 NMS 压不掉）
        self._person_merge_vert_iou_min = 0.64   # 竖向重叠 / min(h1,h2)
        self._person_merge_x_gap_frac = 0.032  # 横向最大间隙 ≈ 画面宽度比例
        self._person_merge_x_gap_min_px = 14.0
        self._person_merge_iou_min = 0.10      # 略重叠即并
        self._person_merge_union_wh_max = 0.78 # 合并后过宽则更像两人并排，需额外条件

        # 跟踪后 ID 去重：同一人若被分成两个 track_id（如半身在区内边界处），
        # 依据「足点接近 + 竖向高度重叠」持续若干帧后判为同一人，归并轨迹。
        self.id_alias: dict = {}            # raw_id -> canonical_id（持久）
        self._dedup_pair_persist: dict = {} # (idA,idB) -> 连续满足帧数
        self._dedup_last_seen: dict = {}    # raw_id -> 最近出现帧，用于清理过期 alias
        self._dedup_persist_frames = 3      # 连续满足几帧才归并，防抖
        self._dedup_alias_ttl = 150         # alias 中的 id 超过该帧数未出现则清除

    # ------------------------------------------------------------------ helpers

    def get_class_color(self, class_id: int):
        name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
        if name == 'person':
            return self.class_colors['person']
        elif name in ('bicycle', 'car', 'motorcycle', 'bus', 'truck'):
            return self.class_colors['vehicle']
        elif name in ('bird', 'cat', 'dog', 'horse', 'sheep', 'cow'):
            return self.class_colors['animal']
        return self.class_colors['other']

    def _should_merge_person_pair(self, bi, bj, frame_w, frame_h):
        """
        判断两个 person 框是否像「同一目标被竖切的两块」：竖向对齐度高且横向可衔接；
        与常规 NMS 互补（并排半框 IoU 往往接近 0）。
        """
        _ = frame_h
        x1i, y1i, x2i, y2i = [float(v) for v in bi[:4]]
        x1j, y1j, x2j, y2j = [float(v) for v in bj[:4]]
        hi = max(y2i - y1i, 1.0)
        hj = max(y2j - y1j, 1.0)
        wi = max(x2i - x1i, 1.0)
        wj = max(x2j - x1j, 1.0)

        iy = max(0.0, min(y2i, y2j) - max(y1i, y1j))
        vy = iy / min(hi, hj)
        if vy < self._person_merge_vert_iou_min:
            return False

        inter = self._rect_intersection_area(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j)
        uni = wi * hi + wj * hj - inter + 1e-6
        iou = inter / uni
        if iou >= self._person_merge_iou_min:
            return True

        gap_x = max(0.0, max(x1i, x1j) - min(x2i, x2j))
        gap_th = max(self._person_merge_x_gap_min_px, self._person_merge_x_gap_frac * frame_w)
        if gap_x > gap_th:
            return False

        # 同一目标被竖切的两块框宽通常接近；成人+小孩并排时宽差大，降低误并
        width_sim = min(wi, wj) / max(wi, wj)
        if iou < self._person_merge_iou_min and width_sim < 0.42:
            return False

        ux1, uy1 = min(x1i, x1j), min(y1i, y1j)
        ux2, uy2 = max(x2i, x2j), max(y2i, y2j)
        uw, uh = max(ux2 - ux1, 1.0), max(uy2 - uy1, 1.0)
        ar_u = uw / uh
        ar_i = wi / hi
        ar_j = wj / hj
        narrow_strip = min(ar_i, ar_j) < 0.42

        if ar_u > self._person_merge_union_wh_max and not narrow_strip:
            return False
        return True

    def _merge_fragment_person_detections(self, detections, frame_w, frame_h):
        """对 person 类做碎片合并，其它类别原样保留。"""
        if not detections or len(detections) < 2:
            return detections

        det = np.asarray(detections, dtype=np.float64)
        if det.ndim != 2 or det.shape[1] < 6:
            return detections

        cls_col = det[:, 5].astype(np.int32)
        person_mask = cls_col == 0
        if int(np.sum(person_mask)) < 2:
            return detections

        p_idx = np.flatnonzero(person_mask)
        o_idx = np.flatnonzero(~person_mask)
        n_p = len(p_idx)

        parent = np.arange(n_p, dtype=np.int32)

        def find(a):
            a = int(a)
            if int(parent[a]) != a:
                parent[a] = find(int(parent[a]))
            return int(parent[a])

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n_p):
            for j in range(i + 1, n_p):
                ia, ja = int(p_idx[i]), int(p_idx[j])
                if self._should_merge_person_pair(det[ia], det[ja], frame_w, frame_h):
                    union(i, j)

        clusters = {}
        for i in range(n_p):
            r = int(find(i))
            clusters.setdefault(r, []).append(int(p_idx[i]))

        merged_rows = []
        for _root, idxs in clusters.items():
            rows = det[np.array(idxs, dtype=np.int32)]
            x1 = float(np.min(rows[:, 0]))
            y1 = float(np.min(rows[:, 1]))
            x2 = float(np.max(rows[:, 2]))
            y2 = float(np.max(rows[:, 3]))
            conf = float(np.max(rows[:, 4]))
            merged_rows.append([x1, y1, x2, y2, conf, 0.0])

        out = merged_rows + det[o_idx].tolist()
        return out

    def _canonical_id(self, tid):
        """顺着 alias 链找到该 track 的归并目标 ID（带路径压缩）。"""
        tid = int(tid)
        seen = []
        while tid in self.id_alias and self.id_alias[tid] != tid:
            seen.append(tid)
            tid = int(self.id_alias[tid])
        for s in seen:
            self.id_alias[s] = tid
        return tid

    def apply_track_dedup(self, tracked_objects):
        """
        跟踪后归并「同一人被分裂成多个 ID」：足点接近且竖向高度重叠，持续数帧→判为同一人。
        返回归并后的列表：每个 canonical_id 仅保留一个外接框（union），conf 取最大。
        """
        persons = []
        others = []
        for obj in tracked_objects:
            if len(obj) < 7:
                others.append(obj)
                continue
            x1, y1, x2, y2, tid, conf, cls = obj[:7]
            if int(cls) != 0:
                others.append(obj)
            else:
                persons.append([float(x1), float(y1), float(x2), float(y2),
                                int(tid), float(conf)])

        # 当前帧出现的成对关系：满足「同一人被竖切」几何条件则累加持久计数，否则清零。
        # 复用 _should_merge_person_pair，天然排除「两人并排」（并排的 union 过宽会被否决）。
        fw = self.frame_w or 1
        fh = self.frame_h or 1
        active_pairs = set()
        n = len(persons)
        for i in range(n):
            xi1, yi1, xi2, yi2, idi, _ = persons[i]
            for j in range(i + 1, n):
                xj1, yj1, xj2, yj2, idj, _ = persons[j]
                if self._should_merge_person_pair(
                        (xi1, yi1, xi2, yi2), (xj1, yj1, xj2, yj2), fw, fh):
                    key = (min(idi, idj), max(idi, idj))
                    active_pairs.add(key)
                    self._dedup_pair_persist[key] = self._dedup_pair_persist.get(key, 0) + 1
                    if self._dedup_pair_persist[key] >= self._dedup_persist_frames:
                        lo, hi_id = key
                        self.id_alias[self._canonical_id(hi_id)] = self._canonical_id(lo)

        for key in list(self._dedup_pair_persist):
            if key not in active_pairs:
                del self._dedup_pair_persist[key]

        # 记录出现帧并清理长期未出现的 alias（防止 ID 复用导致错误归并）
        for _x1, _y1, _x2, _y2, tid, _c in persons:
            self._dedup_last_seen[int(tid)] = self.frame_count
        for rid in list(self.id_alias):
            last = self._dedup_last_seen.get(rid, self.frame_count)
            if self.frame_count - last > self._dedup_alias_ttl:
                self.id_alias.pop(rid, None)
                self._dedup_last_seen.pop(rid, None)

        # 按 canonical_id 归并外接框
        groups: dict = {}
        for x1, y1, x2, y2, tid, conf in persons:
            cid = self._canonical_id(tid)
            g = groups.get(cid)
            if g is None:
                groups[cid] = [x1, y1, x2, y2, conf]
            else:
                g[0] = min(g[0], x1); g[1] = min(g[1], y1)
                g[2] = max(g[2], x2); g[3] = max(g[3], y2)
                g[4] = max(g[4], conf)

        merged = [[g[0], g[1], g[2], g[3], cid, g[4], 0] for cid, g in groups.items()]
        return merged + list(others)

    @staticmethod
    def _rect_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float((ix2 - ix1) * (iy2 - iy1))

    def _person_zone_geometry_score(self, x1, y1, x2, y2):
        """
        单帧「在区内」程度 0~1，基于警戒区多边形 / 地面投影：
        - 足点（脚着地处）是否落在区域内：站位最可信的依据；
        - 下半身带网格点落在区域内的比例：半身进入时显著升高，仅手臂伸入时很低。
        配了地面单应矩阵时，以上判断都在「地面坐标」完成，更接近真实 3D 站位。
        """
        w = self.frame_w or 1
        h = self.frame_h or 1
        ax1, ay1, ax2, ay2 = float(x1), float(y1), float(x2), float(y2)

        foot_x = (ax1 + ax2) * 0.5
        foot_y = ay2
        foot_in = self.zone.point_in_zone(foot_x, foot_y, w, h)

        cover = self.zone.coverage_ratio(ax1, ay1, ax2, ay2, w, h,
                                         lower_band=self._zone_lower_body_ratio)

        score = 0.45 * (1.0 if foot_in else 0.0) + 0.55 * float(np.clip(cover / 0.6, 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0)), bool(foot_in)

    def update_zone_membership(self, tracked_objects):
        """
        每帧根据检测更新各 track 的区内得分 EMA + Schmitt 滞回。
        必须在同一帧内先于绘制与逗留逻辑调用一次。
        """
        seen = set()
        for obj in tracked_objects:
            if len(obj) < 7:
                continue
            x1, y1, x2, y2, track_id, _conf, cls = obj[:7]
            if int(cls) != 0:
                continue
            tid = int(track_id)
            seen.add(tid)
            g, _foot = self._person_zone_geometry_score(x1, y1, x2, y2)
            prev = self._zone_score_ema.get(tid, g)
            ema = self._zone_score_ema_alpha * g + (1.0 - self._zone_score_ema_alpha) * prev
            self._zone_score_ema[tid] = ema
            self._zone_last_seen_frame[tid] = self.frame_count

            prev_h = self._zone_hyst_in.get(tid, False)
            if ema >= self._zone_schmitt_high:
                self._zone_hyst_in[tid] = True
            elif ema <= self._zone_schmitt_low:
                self._zone_hyst_in[tid] = False
            else:
                self._zone_hyst_in[tid] = prev_h

        # 本帧未出现的人：不更新 EMA；长时间未出现则清理，避免 dict 膨胀
        for tid in list(self._zone_score_ema):
            if tid in seen:
                continue
            last_f = self._zone_last_seen_frame.get(tid, self.frame_count)
            if self.frame_count - last_f > 45:
                self._zone_score_ema.pop(tid, None)
                self._zone_hyst_in.pop(tid, None)
                self._zone_last_seen_frame.pop(tid, None)

    def is_track_substantially_in_zone(self, track_id) -> bool:
        """滞回后的在区判定（用于画框、逗留逻辑）。"""
        return bool(self._zone_hyst_in.get(int(track_id), False))

    # ------------------------------------------------------------------ alert logic

    def check_lingering(self, tracked_objects, frame):
        current_time = time.time()
        new_alerts = []
        detected_ids = set()   # 本帧所有检测到的人

        for obj in tracked_objects:
            if len(obj) < 7:
                continue
            x1, y1, x2, y2, track_id, conf, cls = obj[:7]
            if int(cls) != 0:
                continue

            detected_ids.add(track_id)
            in_zone = self.is_track_substantially_in_zone(track_id)

            if in_zone:
                # 在区域内：in_counter +1，out_counter 清零
                self._zone_in_counter[track_id]  = self._zone_in_counter.get(track_id, 0) + 1
                self._zone_out_counter[track_id] = 0

                # 累积够 IN_FRAMES 帧才正式注册
                if self._zone_in_counter[track_id] >= self._ZONE_IN_FRAMES:
                    if track_id not in self.lingering_targets:
                        self.lingering_targets[track_id] = {
                            'enter_time':      current_time,
                            'last_seen':       current_time,
                            'last_alert_time': current_time,
                            'alert_count':     0,
                        }
                        print(f"Target {int(track_id)} entered detection zone")
                    else:
                        info = self.lingering_targets[track_id]
                        info['last_seen'] = current_time

                        linger_time = current_time - info['enter_time']
                        if (linger_time >= self.lingering_threshold and
                                current_time - info['last_alert_time'] >= self.lingering_threshold):
                            alert = self.trigger_lingering_alert(track_id, linger_time, frame)
                            new_alerts.append(alert)
                            info['last_alert_time'] = current_time
                            info['alert_count'] += 1
                            print(f"🚨 Alert #{info['alert_count']}: target {int(track_id)} "
                                  f"lingering {linger_time:.1f}s")
            else:
                # 不在区域内：out_counter +1，in_counter 清零
                self._zone_out_counter[track_id] = self._zone_out_counter.get(track_id, 0) + 1
                self._zone_in_counter[track_id]  = 0

        # 连续 OUT_FRAMES 帧不在区域 → 才真正移除
        to_remove = []
        for tid in list(self.lingering_targets):
            out_frames = self._zone_out_counter.get(tid, 0)
            not_detected = tid not in detected_ids
            if out_frames >= self._ZONE_OUT_FRAMES or (not_detected and
                    current_time - self.lingering_targets[tid]['last_seen'] > 5):
                linger_time = current_time - self.lingering_targets[tid]['enter_time']
                if linger_time > 3:
                    print(f"Target {int(tid)} left zone after {linger_time:.1f}s")
                to_remove.append(tid)
        for tid in to_remove:
            del self.lingering_targets[tid]
            self._zone_in_counter.pop(tid, None)
            self._zone_out_counter.pop(tid, None)

        # 清理已消失目标的计数器
        for tid in list(self._zone_in_counter):
            if tid not in detected_ids:
                self._zone_in_counter.pop(tid, None)
                self._zone_out_counter.pop(tid, None)

        return new_alerts

    def trigger_lingering_alert(self, track_id, linger_time, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lingering_{track_id}_{ts}.jpg"
        filepath = self.screenshots_dir / filename

        alert_frame = frame.copy()
        cv2.putText(alert_frame, f"LINGERING ALERT - ID: {track_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(alert_frame,
                    f"Time: {linger_time:.1f}s - {datetime.now().strftime('%H:%M:%S')}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(filepath), alert_frame)

        alert_info = {
            'timestamp':  datetime.now().isoformat(),
            'track_id':   int(track_id),
            'linger_time': round(linger_time, 1),
            'screenshot': str(filepath),
            'type':       'lingering',
        }
        self._save_alert(alert_info)
        return alert_info

    def _save_alert(self, alert_info):
        alerts = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []
        alerts.append(alert_info)
        alerts = alerts[-100:]      # keep latest 100
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)

    # ------------------------------------------------------------------ trajectory

    def _trajectory_anchor(self, x1, y1, x2, y2):
        """人体底部略上方的画面锚点（监控常用：比纯 y2 更易读、不易贴边）。"""
        ax1, ay1, ax2, ay2 = float(x1), float(y1), float(x2), float(y2)
        bh = max(ay2 - ay1, 1.0)
        lift = max(self.track_anchor_bottom_lift_min_px,
                   self.track_anchor_bottom_lift_ratio * bh)
        lift = min(lift, 0.38 * bh)  # 不高于躯干中段，仍以「脚下区域」为主
        cx = (ax1 + ax2) * 0.5
        cy = ay2 - lift
        return cx, cy

    def update_person_tracks(self, tracked_objects):
        """记录全画面所有人员轨迹锚点（足底略上方，兼顾俯视与平视机位）。"""
        seen_ids = set()
        for obj in tracked_objects:
            if len(obj) < 7:
                continue
            x1, y1, x2, y2, track_id, conf, cls = obj[:7]
            if int(cls) != 0:
                continue

            cx, cy = self._trajectory_anchor(x1, y1, x2, y2)
            seen_ids.add(track_id)
            self.track_miss_counts[track_id] = 0

            if track_id not in self.person_tracks:
                self.person_tracks[track_id] = [(cx, cy)]
                continue

            last_x, last_y = self.person_tracks[track_id][-1]
            # EMA 平滑处理，减少检测框抖动带来的轨迹跳变
            smooth_x = self.track_ema_alpha * cx + (1.0 - self.track_ema_alpha) * last_x
            smooth_y = self.track_ema_alpha * cy + (1.0 - self.track_ema_alpha) * last_y

            # 距离过滤：位移太小（可能是原地踏步或抖动）不增加新点
            if np.hypot(smooth_x - last_x, smooth_y - last_y) >= self.track_min_move_px:
                self.person_tracks[track_id].append((smooth_x, smooth_y))
                if len(self.person_tracks[track_id]) > self.track_max_len:
                    self.person_tracks[track_id].pop(0)

        # 漏检容忍逻辑：如果目标短时间消失（如被遮挡或在框边缘），保留其轨迹，不立即删除 ID
        for tid in list(self.person_tracks):
            if tid in seen_ids:
                continue
            self.track_miss_counts[tid] = self.track_miss_counts.get(tid, 0) + 1
            if self.track_miss_counts[tid] > self.track_keep_missing:
                del self.person_tracks[tid]
                self.track_miss_counts.pop(tid, None)

    def draw_all_trajectories(self, frame):
        """
        商业监控风格尾迹：高对比描边 + 亮度下限，避免旧段乘 alpha 后接近黑色看不见；
        单点目标画圆点，避免「刚出现像没轨迹」。
        """
        for tid, positions in self.person_tracks.items():
            if not positions:
                continue

            in_zone = self.is_track_substantially_in_zone(int(tid))
            base_color = (0, 165, 255) if not in_zone else (0, 0, 255)
            outline = (40, 40, 40)

            if len(positions) < 2:
                lx, ly = int(positions[-1][0]), int(positions[-1][1])
                cv2.circle(frame, (lx, ly), 5, outline, -1, cv2.LINE_AA)
                cv2.circle(frame, (lx, ly), 3, base_color, -1, cv2.LINE_AA)
                continue

            n = len(positions)
            denom = max(n - 1, 1)
            for i in range(1, n):
                p1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
                p2 = (int(positions[i][0]), int(positions[i][1]))
                # recency: 靠近当前帧的线段更亮；旧段仍保持最低约 55% 亮度（不再用接近 0 的乘子）
                recency = i / float(denom)
                dim = tuple(int(0.42 * c) for c in base_color)
                color = tuple(
                    int(dim[k] + (base_color[k] - dim[k]) * recency) for k in range(3)
                )
                thickness = max(2, int(round(2 + 3 * recency)))
                cv2.line(frame, p1, p2, outline, thickness + 2, cv2.LINE_AA)
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ drawing

    def draw_detection_zone(self, frame):
        h, w = frame.shape[:2]
        poly = self.zone.image_polygon(w, h).astype(np.int32)
        if len(poly) < 3:
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 255, 0), 3, cv2.LINE_AA)
        tx, ty = int(poly[:, 0].min()), int(poly[:, 1].min())
        tag = "DETECTION ZONE (3D)" if self.zone.homography is not None else "DETECTION ZONE"
        cv2.putText(frame, tag, (tx + 5, ty + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Lingering >{int(self.lingering_threshold)}s triggers alert",
                    (tx + 5, ty + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_lingering_info(self, frame):
        current_time = time.time()
        for track_id, info in self.lingering_targets.items():
            linger_time = current_time - info['enter_time']
            if linger_time <= 3:
                continue

            # 轨迹已经在 draw_all_trajectories 中统一绘制
            # 这里只负责显示时间标签和警报次数
            if track_id in self.person_tracks and self.person_tracks[track_id]:
                lx, ly = self.person_tracks[track_id][-1]
                text = f"{linger_time:.1f}s"
                tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame,
                              (int(lx) - 5, int(ly) - th - 30),
                              (int(lx) + tw + 5, int(ly) - 25), (0, 0, 0), -1)
                color = (0, 165, 255) if linger_time < 15 else (0, 0, 255)
                cv2.putText(frame, text, (int(lx), int(ly) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if info['alert_count'] > 0:
                    at = f"Alerts: {info['alert_count']}"
                    atw, ath = cv2.getTextSize(at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame,
                                  (int(lx) - 5, int(ly) - ath - 60),
                                  (int(lx) + atw + 5, int(ly) - 55), (0, 0, 0), -1)
                    cv2.putText(frame, at, (int(lx), int(ly) - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ------------------------------------------------------------------ main loop

    def process_camera(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        ret, first_frame = cap.read()
        if not ret:
            print("❌ Cannot read camera frame")
            return

        h, w = first_frame.shape[:2]
        self.frame_w, self.frame_h = w, h
        # 保存一帧供 Web 后台「警戒区配置」页作为底图绘制
        try:
            cv2.imwrite(str(self.data_dir / 'last_frame.jpg'), first_frame)
        except Exception:
            pass
        print(f"Zone polygon points: {len(self.zone.polygon_norm)} | "
              f"ground-homography: {'on' if self.zone.homography is not None else 'off'}")
        print("🚀 Starting real-time monitoring — press 'q' to quit, 's' to screenshot")

        prev_time = time.time()
        fps_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame read error")
                break

            self.frame_count += 1
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps_buffer.append(1.0 / dt)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
            prev_time = current_time
            avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

            self.frame_h, self.frame_w = frame.shape[:2]
            # 热加载：Web 后台改了警戒区即时生效；每 ~1s 存一帧底图供后台编辑
            if self.frame_count % 15 == 0:
                if self.zone.load():
                    print("🔄 警戒区配置已更新")
                try:
                    cv2.imwrite(str(self.data_dir / 'last_frame.jpg'), frame)
                except Exception:
                    pass

            self.draw_detection_zone(frame)

            # iou 略收紧，减轻重复框；并排「半人」框仍靠下方碎片合并
            results = self.yolo_model(frame, verbose=False, conf=0.3, iou=0.48)
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2,
                                           box.conf[0].cpu().numpy(),
                                           box.cls[0].cpu().numpy()])

            fh, fw = frame.shape[:2]
            detections = self._merge_fragment_person_detections(detections, fw, fh)

            if detections:
                try:
                    tracked = self.tracker.update(np.array(detections), frame)
                    tracked = self.apply_track_dedup(tracked)  # 同一人多 ID → 归并
                    self.update_zone_membership(tracked)
                    self.update_person_tracks(tracked)
                    self.check_lingering(tracked, frame)

                    for obj in tracked:
                        if len(obj) < 7:
                            continue
                        x1, y1, x2, y2, track_id, conf, cls = obj[:7]
                        in_zone = self.is_track_substantially_in_zone(track_id)
                        color = (0, 255, 0) if in_zone else self.get_class_color(int(cls))
                        thickness = 3 if in_zone else 2
                        cv2.rectangle(frame,
                                      (int(x1), int(y1)), (int(x2), int(y2)),
                                      color, thickness)
                        label = f"ID:{int(track_id)} {conf:.2f}"
                        if in_zone:
                            label += " [IN ZONE]"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    print(f"Tracking error: {e}")
            else:
                # 本帧无检测时仍衰减轨迹缓存，避免「人走开一瞬间尾迹整块消失」
                self.update_person_tracks([])

            # 尾迹画在检测框之上，商业回放里路径更清晰
            self.draw_all_trajectories(frame)
            self.draw_lingering_info(frame)

            elapsed = time.time() - self.start_time
            active = len(self.lingering_targets)
            total_alerts = sum(i['alert_count'] for i in self.lingering_targets.values())
            for y_pos, text, color in [
                (30, "Smart Security - Zone-Based Lingering Detection", (0, 255, 0)),
                (50, f"FPS: {avg_fps:.1f} | Frame: {self.frame_count}", (255, 255, 255)),
                (70, f"Targets: {active} | Alerts: {total_alerts}", (0, 0, 255)),
                (90, f"Run time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}", (255, 255, 255)),
            ]:
                cv2.putText(frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow('Smart Security - Zone-Based Lingering Alert', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                fname = f"manual_screenshot_{ts}.jpg"
                cv2.imwrite(str(self.screenshots_dir / fname), frame)
                print(f"📸 Screenshot saved: {fname}")

        cap.release()
        cv2.destroyAllWindows()
        total_time = time.time() - self.start_time
        print(f"\nFinal stats — frames: {self.frame_count} | "
              f"avg FPS: {self.frame_count / total_time:.1f}")
        print("✅ System shutdown complete")


if __name__ == "__main__":
    MultiObjectTracker().process_camera(0)