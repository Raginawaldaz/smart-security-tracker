# 智能安防追踪系统（smart-security-tracker）

基于 **YOLOv8** 检测与 **StrongSORT**（BoxMOT）跟踪的区域逗留报警系统：在可编辑的警戒多边形内统计人员停留时间，超时后截图告警，并通过 Streamlit 后台查看记录、编辑警戒区。

Python 包名：`smart_security`。推荐仓库名：`smart-security-tracker`。

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

配置加载优先级（高 → 低）：操作系统环境变量 → `.env` → `config.yaml` → 代码默认值。业务代码只从 `get_settings()` 取值。

## 项目结构

```text
main.py
src/smart_security/
  core/           # 配置、数据库、异常
  modules/
    tracking/     # 检测、碎片合并、ID 归并、轨迹、主循环
    zone/         # 警戒区配置与区内判定
    alarm/        # 逗留状态机与落盘
    dashboard/    # Streamlit 页面
  utils/          # 日志、torch 补丁
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

## 仓库改名

若将 GitHub 仓库从 `my_yolov_deepsort_project` 改为 `smart-security-tracker`：在仓库 Settings 中改名后执行

```bash
git remote set-url origin https://github.com/Raginawaldaz/smart-security-tracker.git
```

本项目规范要求使用**私有仓库**托管。

## 许可证

AGPL-3.0。本项目仅用于学习与实验。
