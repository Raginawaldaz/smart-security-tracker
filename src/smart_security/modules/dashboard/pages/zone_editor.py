# 模块：警戒区可视化编辑页
# 作用：在摄像头底图上拖拽多边形/矩形，保存后由监控端按 mtime 热加载。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from smart_security.core.config import get_settings
from smart_security.modules.zone.zone_config import ZoneConfig, default_polygon_norm
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

CANVAS_DISPLAY_WIDTH = 760
HOMOGRAPHY_POINT_COUNT = 4
EXPANSION_SLIDER_MAX = 60


def _parse_canvas_polygon(
    objects: list[dict[str, Any]], display_width: int, display_height: int
) -> list[list[float]] | None:
    if not objects:
        return None
    shape = objects[-1]
    points: list[list[float]] = []
    shape_type = shape.get("type")
    if shape_type == "rect":
        left = shape.get("left", 0)
        top = shape.get("top", 0)
        width = shape.get("width", 0) * shape.get("scaleX", 1)
        height = shape.get("height", 0) * shape.get("scaleY", 1)
        points = [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ]
    elif shape_type in ("path", "polygon"):
        raw = shape.get("path") or shape.get("points")
        if raw:
            for segment in raw:
                if isinstance(segment, dict) and "x" in segment:
                    points.append([segment["x"], segment["y"]])
                elif isinstance(segment, (list, tuple)):
                    numbers = [
                        value for value in segment if isinstance(value, (int, float))
                    ]
                    if len(numbers) >= 2:
                        points.append([numbers[-2], numbers[-1]])
    if len(points) < 3:
        return None
    return [
        [
            max(0.0, min(1.0, x / display_width)),
            max(0.0, min(1.0, y / display_height)),
        ]
        for x, y in points
    ]


def _capture_frame(frame_path: Path) -> None:
    import cv2

    settings = get_settings()
    capture = cv2.VideoCapture(settings.camera.source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        st.error("无法从摄像头抓拍，请确认摄像头未被监控端占用。")
        return
    cv2.imwrite(str(frame_path), frame)
    logger.info("Captured preview frame to %s", frame_path)
    st.success("已抓拍底图。")
    st.rerun()


def _homography_editor(
    zone: ZoneConfig, original_width: int, original_height: int
) -> None:
    import cv2

    st.write(
        f"图像点以原始分辨率像素填写（当前底图：{original_width}×{original_height}）。"
    )
    image_points: list[list[float]] = []
    ground_points: list[list[float]] = []
    columns = st.columns(HOMOGRAPHY_POINT_COUNT)
    for index in range(HOMOGRAPHY_POINT_COUNT):
        with columns[index]:
            st.markdown(f"**点 {index + 1}**")
            image_x = st.number_input(f"img_x{index}", value=0.0, key=f"ix{index}")
            image_y = st.number_input(f"img_y{index}", value=0.0, key=f"iy{index}")
            ground_x = st.number_input(f"gnd_x{index}", value=0.0, key=f"gx{index}")
            ground_y = st.number_input(f"gnd_y{index}", value=0.0, key=f"gy{index}")
            image_points.append([image_x, image_y])
            ground_points.append([ground_x, ground_y])

    col_enable, col_disable = st.columns(2)
    with col_enable:
        if st.button("计算并启用地面投影", use_container_width=True):
            source = np.array(image_points, dtype=np.float64)
            dest = np.array(ground_points, dtype=np.float64)
            if np.allclose(source, 0) or np.allclose(dest, 0):
                st.warning("请填入有效的 4 组对应点。")
            else:
                homography, _mask = cv2.findHomography(source, dest)
                if homography is None:
                    st.error("单应矩阵求解失败，请检查点位（勿共线）。")
                else:
                    zone.homography = homography
                    ground_polygon: list[list[float]] = []
                    for norm_x, norm_y in zone.polygon_norm:
                        vector = homography @ np.array(
                            [norm_x * original_width, norm_y * original_height, 1.0]
                        )
                        if abs(vector[2]) > 1e-9:
                            ground_polygon.append(
                                [
                                    float(vector[0] / vector[2]),
                                    float(vector[1] / vector[2]),
                                ]
                            )
                    zone.ground_polygon = (
                        np.array(ground_polygon) if len(ground_polygon) >= 3 else None
                    )
                    zone.save()
                    st.success("已启用地面投影（3D）判定。")
    with col_disable:
        if st.button("关闭地面投影（回到 2D）", use_container_width=True):
            zone.homography = None
            zone.ground_polygon = None
            zone.save()
            st.success("已切回 2D 多边形判定。")


def render() -> None:
    st.header("警戒区配置（可拖拽）")
    st.caption("在画面上拖拽绘制覆盖范围；保存后监控端会自动热加载，无需重启。")

    try:
        from PIL import Image
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("缺少依赖：请先安装 streamlit-drawable-canvas 与 pillow")
        return

    settings = get_settings()
    zone = ZoneConfig()
    frame_path = settings.last_frame_path

    col_canvas, col_controls = st.columns([3, 1])
    with col_controls:
        if st.button("从摄像头抓拍底图", use_container_width=True):
            _capture_frame(frame_path)
        mode = st.radio(
            "绘制方式",
            ["polygon", "rect"],
            help="polygon 可贴合透视地面；rect 为快速矩形",
        )
        expansion = st.slider(
            "边界外扩(px)", 0, EXPANSION_SLIDER_MAX, int(zone.expansion_px)
        )

    if not frame_path.exists():
        with col_canvas:
            st.info("暂无底图。请先运行监控端生成 last_frame.jpg，或点击右侧抓拍。")
        return

    background = Image.open(str(frame_path)).convert("RGB")
    original_width, original_height = background.size
    display_height = int(original_height * CANVAS_DISPLAY_WIDTH / original_width)
    background_display = background.resize((CANVAS_DISPLAY_WIDTH, display_height))

    with col_canvas:
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.25)",
            stroke_color="#00FF00",
            stroke_width=2,
            background_image=background_display,
            update_streamlit=True,
            height=display_height,
            width=CANVAS_DISPLAY_WIDTH,
            drawing_mode=mode,
            key="zone_canvas",
        )

    objects = (canvas.json_data or {}).get("objects", []) if canvas else []
    new_polygon = _parse_canvas_polygon(objects, CANVAS_DISPLAY_WIDTH, display_height)

    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button(
            "保存警戒区",
            type="primary",
            use_container_width=True,
            disabled=new_polygon is None,
        ):
            zone.polygon_norm = new_polygon
            zone.expansion_px = float(expansion)
            zone.save()
            st.success(f"已保存 {len(new_polygon)} 点多边形，监控端将自动生效。")
    with col_reset:
        if st.button("恢复默认（左半屏）", use_container_width=True):
            zone.polygon_norm = default_polygon_norm()
            zone.expansion_px = float(expansion)
            zone.save()
            st.success("已恢复默认区域。")

    st.markdown("**当前已保存的归一化坐标：**")
    st.code(json.dumps(zone.polygon_norm, ensure_ascii=False, indent=2))

    with st.expander("地面投影（3D）标定 — 进阶可选"):
        st.caption(
            "输入 4 组『图像像素 → 地面坐标(米)』对应点，用于把脚的位置投影到地面平面判断，"
            "更适合高位俯视相机。留空则使用 2D 多边形判定。"
        )
        _homography_editor(zone, original_width, original_height)
