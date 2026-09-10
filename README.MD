# YOLOv5 DeepSORT 项目

本项目是一个基于 YOLOv5 和 DeepSORT 的多目标跟踪系统，结合了 BoxMOT 库提供的先进跟踪算法，实现了高效、准确的目标检测和跟踪功能。

## 📋 项目简介

该项目利用 YOLOv5 进行目标检测，并使用 DeepSORT 及其变体算法进行目标跟踪，适用于视频监控、交通管理、人流统计等场景。项目集成了 BoxMOT 库，提供了多种先进的跟踪算法选择，如 BoostTrack、BoTSORT、ByteTrack 等。

## 🚀 主要功能

- **高效目标检测**：使用 YOLOv5 模型进行实时目标检测
- **多种跟踪算法**：集成 BoxMOT 库，支持多种 SOTA 跟踪算法
- **外观特征匹配**：结合 ReID 模型，提高跨帧目标识别准确率
- **灵活配置**：支持多种模型和参数配置，适应不同场景需求
- **多源输入**：支持摄像头、视频文件、图片等多种输入源

## 🔧 安装说明

### 环境要求
- Python >= 3.9
- PyTorch >= 1.10
- CUDA 支持（可选，用于 GPU 加速）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/yourusername/my_yolov5_deepsort_project.git
   cd my_yolov5_deepsort_project
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **安装 BoxMOT**
   ```bash
   pip install boxmot
   ```

## 💻 使用方法

### 命令行界面

项目提供了基于 BoxMOT 的命令行工具，可用于跟踪、评估和模型导出等操作：

```bash
# 运行跟踪
boxmot track --source 0 --yolo-model yolov5s.pt --tracking-method deepocsort

# 评估跟踪性能
boxmot eval --yolo-model yolov5s.pt --tracking-method deepocsort --source ./assets/MOT17-mini/train

# 导出 ReID 模型
boxmot export --weights weights/osnet_x0_25_msmt17.pt --include onnx
```

### Python API

也可以直接在 Python 代码中集成跟踪功能：

```python
import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import DeepOCSort
from models.common import DetectMultiBackend

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 YOLOv5 模型
model = DetectMultiBackend('yolov5s.pt', device=device)

# 初始化跟踪器
tracker = DeepOCSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device)

# 开始视频捕获
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # 运行目标检测
    results = model(frame)
    detections = results.pred[0].cpu().numpy()
    
    # 过滤低置信度检测结果
    detections = detections[detections[:, 4] > 0.5]
    
    # 更新跟踪器
    # 输入格式: M X (x, y, x, y, conf, cls)
    # 输出格式: M X (x, y, x, y, id, conf, cls, ind)
    tracked_objects = tracker.update(detections, frame)
    
    # 绘制跟踪结果
    tracker.plot_results(frame, show_trajectories=True)
    
    # 显示结果
    cv2.imshow('YOLOv5 DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()
```

## 📁 项目结构

```
my_yolov5_deepsort_project/
├── boxmot/             # BoxMOT 库
│   ├── appearance/     # 外观特征提取模型
│   ├── configs/        # 配置文件
│   ├── engine/         # 核心引擎
│   ├── motion/         # 运动模型
│   └── README.md       # BoxMOT 文档
├── __pycache__/        # 编译缓存
├── .venv/              # 虚拟环境
├── .gitignore          # Git 忽略文件
└── README.MD           # 项目说明文档
```

## 🎯 跟踪算法选择

项目支持多种跟踪算法，可根据场景需求选择：

| 跟踪算法 | 状态 | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| boosttrack | ✅ | 69.253 | 75.914 | 83.206 | 25 |
| botsort | ✅ | 68.885 | 78.222 | 81.344 | 46 |
| hybridsort | ✅ | 68.216 | 76.382 | 81.164 | 25 |
| strongsort | ✅ | 68.05 | 76.185 | 80.763 | 17 |
| deepocsort | ✅ | 67.796 | 75.868 | 80.514 | 12 |
| bytetrack | ✅ | 67.68 | 78.039 | 79.157 | 1265 |
| ocsort | ✅ | 66.441 | 74.548 | 77.899 | 1483 |

## 📚 参考资料

- [BoxMOT: Pluggable SOTA multi-object tracking modules](https://github.com/mikel-brostrom/boxmot)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [DeepSORT](https://github.com/nwojke/deep_sort)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📄 许可证

本项目基于 AGPL 3.0 许可证。

## ⚠️ 项目局限

- **轨迹绘制**：无法准确绘测人物轨迹
- **3D 识别**：目前无法进行 3D 立体的识别人物

## 📝 使用声明

本项目仅用于学习和实验目的，请勿用于商业用途或其他非法用途。

