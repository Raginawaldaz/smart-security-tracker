import yaml
from pathlib import Path


class ConfigManager:
    def __init__(self):
        self.config_path = Path('config/system_config.yaml')
        self.config = self.load_config()

    def load_config(self):
        """加载系统配置"""
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"❌ 加载配置文件失败: {e}")
                return {}
        else:
            # 创建默认配置
            default_config = {
                'camera': {
                    'source': 0,
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'detection': {
                    'confidence_threshold': 0.5,
                    'classes': [0],  # 0: person
                    'model': 'yolov8n.pt'
                },
                'tracking': {
                    'max_age': 30,
                    'min_hits': 3,
                    'iou_threshold': 0.3
                },
                'alarm': {
                    'sound_enabled': True,
                    'save_screenshots': True,
                    'screenshot_quality': 85
                },
                'system': {
                    'log_level': 'INFO',
                    'auto_start': False,
                    'theme': 'dark'
                }
            }
            self.save_config(default_config)
            return default_config

    def save_config(self, config):
        """保存系统配置"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("✅ 系统配置已保存")

    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key, value):
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self.save_config(self.config)