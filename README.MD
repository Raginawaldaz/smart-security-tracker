# 智能安防追踪系统（smart-security-tracker）

基于 **YOLOv8** 检测与 **StrongSORT**（BoxMOT）跟踪的区域逗留报警系统：在可编辑的警戒多边形内统计人员停留时间，超时后截图告警，并通过 Streamlit 后台查看记录、编辑警戒区。

Python 包名：`smart_security`。

## 主要功能

- 摄像头实时检测与多目标跟踪（人员为主）
- 警戒区多边形热加载（Web 后台拖拽后无需重启监控端）
- 可选地面单应，用脚的地面投影做区内判定
- 逗留超时截图告警，JSON / SQLite 落盘
- Streamlit 仪表板：总览、统计、警戒区编辑、截图清理

## 环境要求

- Python >= 3.10
- PyTorch（可选 CUDA）

## 安装

```bash
git clone https://github.com/Raginawaldaz/smart-security-tracker.git
cd smart-security-tracker

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
# 开发 / 测试
pip install -r requirements-dev.txt
pre-commit install
```

CUDA 版 PyTorch 请按官方指引从对应 index 安装，例如 cu130：

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

将 `.env.example` 复制为 `.env` 后按需覆盖敏感或部署相关项。非敏感阈值在 `config.yaml`。

权重默认路径：

- 检测：`models/yolov8n.pt`
- ReID：`weights/osnet_x0_25_msmt17.pt`

## 使用方法

```bash
# 实时监控（默认摄像头 0）
python main.py track
python main.py track --camera 0
python main.py track --camera path/to/video.mp4

# Streamlit 后台
python main.py dashboard

# 初始化数据库；示例数据必须显式 --seed
python main.py init-db
python main.py init-db --seed
```

监控窗口快捷键：`q` 退出，`s` 手动截图。

### 离线评测与跟踪器对比

调参、换跟踪器或换模型前，先在标注视频上跑评测，确认瓶颈在哪里：

```bash
python scripts/evaluate.py --video eval/videos/a.mp4 --labels eval/a.json --zone eval/a_zone.json
python scripts/evaluate.py --video eval/videos/a.mp4 --labels eval/a.json \
    --trackers strongsort botsort deepocsort boosttrack --output reports/a.json
```

标注格式见 `eval/labels.example.json`：每条事件是一个人在警戒区内的连续停留区间（视频秒）。报告包含报警准确率/召回率/延迟、每个事件内出现的区内 ID 数（大于 1 即 ID 断裂）、重关联与归并次数、处理 FPS 和各阶段耗时。选定的跟踪器写入 `config.yaml` 的 `tracking.tracker`。

配置加载优先级（高 → 低）：操作系统环境变量 → `.env` → `config.yaml` → 代码默认值。业务代码只从 `get_settings()` 取值。

## 项目结构

```text
main.py
src/smart_security/
  core/           # 配置、数据库、异常
  modules/
    tracking/     # 检测、跟踪器工厂、断链重关联、碎片合并、ID 归并、轨迹、主循环
    zone/         # 警戒区配置与区内判定
    alarm/        # 逗留状态机与落盘
    evaluation/   # 离线评测指标与执行
    dashboard/    # Streamlit 页面
  utils/          # 日志、torch 补丁、后台写盘
scripts/          # 离线评测入口
eval/             # 标注示例
tests/
config.yaml
.env.example
```

## 开发检查

提交前在项目根目录执行：

```bash
black .
ruff check .
pytest
```

跟踪算法来自 PyPI 上的 [BoxMOT](https://github.com/mikel-brostrom/boxmot)（`boxmot==15.0.9`），不要再把该仓库整份拷进本项目。

## 许可证

AGPL-3.0。本项目仅用于学习与实验。
