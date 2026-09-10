import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path


class AlarmManager:
    def __init__(self):
        self.alarm_zones = []  # 存储报警区域
        self.alarm_history = []  # 存储报警历史
        self.lingering_targets = {}  # 存储滞留目标信息
        self.load_alarm_zones()

    def load_alarm_zones(self):
        """加载报警区域配置"""
        config_file = Path('config/alarm_zones.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.alarm_zones = json.load(f)
                print(f"✅ 加载了 {len(self.alarm_zones)} 个报警区域")
            except Exception as e:
                print(f"❌ 加载报警区域失败: {e}")
                self.alarm_zones = []
        else:
            # 创建默认配置
            default_zones = [
                {
                    "id": 1,
                    "name": "入口警戒区",
                    "zone_type": "rectangle",
                    "coordinates": [100, 100, 300, 300],  # x1, y1, x2, y2
                    "alarm_type": "intrusion",  # intrusion-入侵, lingering-滞留
                    "linger_time": 10,  # 滞留时间阈值（秒）
                    "enabled": True
                }
            ]
            self.save_alarm_zones(default_zones)
            self.alarm_zones = default_zones

    def save_alarm_zones(self, zones):
        """保存报警区域配置"""
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / 'alarm_zones.json', 'w') as f:
            json.dump(zones, f, indent=4)
        print("✅ 报警区域配置已保存")

    def add_alarm_zone(self, zone_config):
        """添加报警区域"""
        zone_config['id'] = len(self.alarm_zones) + 1
        self.alarm_zones.append(zone_config)
        self.save_alarm_zones(self.alarm_zones)
        return zone_config['id']

    def check_intrusion(self, track_id, bbox, frame):
        """检查区域入侵"""
        alarms = []
        for zone in self.alarm_zones:
            if not zone['enabled']:
                continue

            if zone['zone_type'] == 'rectangle' and zone['alarm_type'] == 'intrusion':
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                zx1, zy1, zx2, zy2 = zone['coordinates']
                if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                    # 触发入侵报警
                    alarm_info = {
                        'timestamp': datetime.now().isoformat(),
                        'zone_id': zone['id'],
                        'zone_name': zone['name'],
                        'track_id': track_id,
                        'alarm_type': 'intrusion',
                        'bbox': bbox
                    }
                    alarms.append(alarm_info)

                    # 在画面上绘制报警区域
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ALARM: {zone['name']}", (zx1, zy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return alarms

    def check_lingering(self, track_id, bbox, frame):
        """检查人员滞留"""
        alarms = []
        current_time = time.time()

        for zone in self.alarm_zones:
            if not zone['enabled'] or zone['alarm_type'] != 'lingering':
                continue

            if zone['zone_type'] == 'rectangle':
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                zx1, zy1, zx2, zy2 = zone['coordinates']
                if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                    # 目标在滞留区域内
                    key = f"{track_id}_{zone['id']}"

                    if key not in self.lingering_targets:
                        # 新目标进入区域
                        self.lingering_targets[key] = {
                            'enter_time': current_time,
                            'last_seen': current_time
                        }
                    else:
                        # 更新最后出现时间
                        self.lingering_targets[key]['last_seen'] = current_time

                        # 检查是否超时
                        linger_time = current_time - self.lingering_targets[key]['enter_time']
                        if linger_time >= zone['linger_time']:
                            alarm_info = {
                                'timestamp': datetime.now().isoformat(),
                                'zone_id': zone['id'],
                                'zone_name': zone['name'],
                                'track_id': track_id,
                                'alarm_type': 'lingering',
                                'linger_time': linger_time,
                                'bbox': bbox
                            }
                            alarms.append(alarm_info)

                            # 绘制报警信息
                            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
                            cv2.putText(frame, f"LINGERING: {zone['name']}", (zx1, zy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # 目标离开区域，移除记录
                    key = f"{track_id}_{zone['id']}"
                    if key in self.lingering_targets:
                        del self.lingering_targets[key]

        return alarms

    def process_alarms(self, tracked_objects, frame):
        """处理所有报警"""
        all_alarms = []

        for obj in tracked_objects:
            if len(obj) >= 7:
                x1, y1, x2, y2, track_id, conf, cls = obj[:7]
                bbox = [x1, y1, x2, y2]

                # 检查入侵报警
                intrusion_alarms = self.check_intrusion(track_id, bbox, frame)
                all_alarms.extend(intrusion_alarms)

                # 检查滞留报警
                lingering_alarms = self.check_lingering(track_id, bbox, frame)
                all_alarms.extend(lingering_alarms)

        return all_alarms

    def save_alarm_screenshot(self, frame, alarm_info):
        """保存报警截图"""
        screenshots_dir = Path('data/screenshots')
        screenshots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alarm_{alarm_info['alarm_type']}_{alarm_info['track_id']}_{timestamp}.jpg"
        filepath = screenshots_dir / filename

        # 在图像上添加报警信息
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame,
                    f"{alarm_info['alarm_type']} - {alarm_info['zone_name']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame,
                    f"Track ID: {alarm_info['track_id']} - Time: {alarm_info['timestamp']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(str(filepath), annotated_frame)
        return str(filepath)