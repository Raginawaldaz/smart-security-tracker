# 模块：警戒区可视化编辑页
# 作用：在摄像头底图上拖拽多边形/矩形，并用安装高度/俯仰角启用地面 3D 判定。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from smart_security.core.config import get_settings
from smart_security.modules.zone.camera import (
    DEFAULT_HEIGHT_M,
    DEFAULT_HFOV_DEG,
    DEFAULT_PITCH_DEG,
    DEFAULT_ROLL_DEG,
    MAX_CAMERA_HEIGHT_M,
    MAX_HFOV_DEG,
    MAX_PITCH_DEG,
    MAX_ROLL_DEG,
    MIN_CAMERA_HEIGHT_M,
    MIN_HFOV_DEG,
    MIN_PITCH_DEG,
    MIN_ROLL_DEG,
    ground_grid_segments,
    image_to_ground_homography,
)
from smart_security.modules.zone.zone_config import ZoneConfig, default_polygon_norm
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

CANVAS_DISPLAY_WIDTH = 760
HOMOGRAPHY_POINT_COUNT = 4
EXPANSION_SLIDER_MAX = 60
GRID_COLOR = (0, 180, 255)
GRID_ORIGIN_COLOR = (0, 255, 255)


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


def _overlay_ground_grid(
    image: np.ndarray,
    homography: np.ndarray,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    import cv2

    display_h, display_w = image.shape[:2]
    scale_x = display_w / max(source_width, 1)
    scale_y = display_h / max(source_height, 1)
    canvas = image.copy()
    for (x1, y1), (x2, y2) in ground_grid_segments(
        homography, source_width, source_height
    ):
        start = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        end = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
        cv2.line(canvas, start, end, GRID_COLOR, 1, cv2.LINE_AA)
    origin = _ground_to_display(
        homography, 0.0, 0.0, scale_x, scale_y, display_w, display_h
    )
    forward = _ground_to_display(
        homography, 0.0, 2.0, scale_x, scale_y, display_w, display_h
    )
    if origin is not None and forward is not None:
        cv2.arrowedLine(canvas, origin, forward, GRID_ORIGIN_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Y+ 2m",
            (forward[0] + 4, forward[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GRID_ORIGIN_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _ground_to_display(
    homography: np.ndarray,
    east: float,
    north: float,
    scale_x: float,
    scale_y: float,
    display_w: int,
    display_h: int,
) -> tuple[int, int] | None:
    from smart_security.modules.zone.camera import project_ground_to_pixels

    pixel = project_ground_to_pixels(homography, np.array([[east, north]]))[0]
    if not np.isfinite(pixel).all():
        return None
    x, y = int(round(pixel[0] * scale_x)), int(round(pixel[1] * scale_y))
    if not (-50 <= x < display_w + 50 and -50 <= y < display_h + 50):
        return None
    return x, y


def _apply_camera_ground(
    zone: ZoneConfig,
    width: int,
    height: int,
    height_m: float,
    pitch_deg: float,
    hfov_deg: float,
    roll_deg: float,
) -> bool:
    zone.camera_height_m = float(height_m)
    zone.pitch_deg = float(pitch_deg)
    zone.hfov_deg = float(hfov_deg)
    zone.roll_deg = float(roll_deg)
    if not zone.enable_camera_ground(width, height):
        st.error("地面单应计算失败，请检查高度/俯仰/视场角。")
        return False
    zone.save()
    return True


def _camera_geometry_controls(
    zone: ZoneConfig,
) -> tuple[bool, float, float, float, float]:
    enabled = st.checkbox(
        "启用地面 3D（按脚底判定）",
        value=zone.ground_source == "camera",
        help="按摄像头安装高度和俯仰把区域投到地面。身体探进画面绿区但脚在区外时不算进入。",
    )
    height_m = st.slider(
        "安装高度 (m)",
        MIN_CAMERA_HEIGHT_M,
        MAX_CAMERA_HEIGHT_M,
        float(zone.camera_height_m or DEFAULT_HEIGHT_M),
        0.1,
    )
    pitch_deg = st.slider(
        "俯仰角 (°，低头为正)",
        MIN_PITCH_DEG,
        MAX_PITCH_DEG,
        float(zone.pitch_deg or DEFAULT_PITCH_DEG),
        1.0,
    )
    hfov_deg = st.slider(
        "水平视场角 (°)",
        MIN_HFOV_DEG,
        MAX_HFOV_DEG,
        float(zone.hfov_deg or DEFAULT_HFOV_DEG),
        1.0,
    )
    roll_deg = st.slider(
        "滚转 (°)",
        MIN_ROLL_DEG,
        MAX_ROLL_DEG,
        float(zone.roll_deg or DEFAULT_ROLL_DEG),
        0.5,
    )
    return enabled, height_m, pitch_deg, hfov_deg, roll_deg


def _homography_editor(
    zone: ZoneConfig, original_width: int, original_height: int
) -> None:
    import cv2

    st.write(
        f"图像点以原始分辨率像素填写（当前底图：{original_width}×{original_height}）。"
        "启用后会关闭相机几何标定。"
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
        if st.button("用 4 点启用地面投影", use_container_width=True):
            source = np.array(image_points, dtype=np.float64)
            dest = np.array(ground_points, dtype=np.float64)
            if np.allclose(source, 0) or np.allclose(dest, 0):
                st.warning("请填入有效的 4 组对应点。")
            else:
                homography, _mask = cv2.findHomography(source, dest)
                if homography is None:
                    st.error("单应矩阵求解失败，请检查点位（勿共线）。")
                elif zone.enable_point_ground(
                    homography, original_width, original_height
                ):
                    zone.save()
                    st.success("已启用 4 点地面投影，相机几何标定已关闭。")
                    st.rerun()
                else:
                    st.error("地面多边形投影失败。")
    with col_disable:
        if st.button(
            "关闭地面投影（回到 2D）", use_container_width=True, key="off_pts"
        ):
            zone.disable_ground_3d()
            zone.save()
            st.success("已切回 2D 多边形判定。")
            st.rerun()


def render() -> None:
    st.header("警戒区配置（可拖拽）")
    st.caption(
        "先填摄像头安装参数启用地面 3D，再在画面上画区域。"
        "保存后监控端按脚是否落在地面多边形内判定，无需重启。"
    )

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
            "边界外扩(px，仅 2D)", 0, EXPANSION_SLIDER_MAX, int(zone.expansion_px)
        )
        st.markdown("**地面 3D 标定**")
        enabled, height_m, pitch_deg, hfov_deg, roll_deg = _camera_geometry_controls(
            zone
        )

    if not frame_path.exists():
        with col_canvas:
            st.info("暂无底图。请先运行监控端生成 last_frame.jpg，或点击右侧抓拍。")
        return

    background = Image.open(str(frame_path)).convert("RGB")
    original_width, original_height = background.size
    display_height = int(original_height * CANVAS_DISPLAY_WIDTH / original_width)

    preview_h = None
    if enabled:
        try:
            preview_h = image_to_ground_homography(
                original_width,
                original_height,
                height_m,
                pitch_deg,
                hfov_deg,
                roll_deg,
            )
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            preview_h = None
    elif zone.uses_ground_plane and zone.homography is not None:
        preview_h = zone.homography

    background_np = np.array(background)
    if preview_h is not None:
        background_np = _overlay_ground_grid(
            background_np, preview_h, original_width, original_height
        )
    background_display = Image.fromarray(background_np).resize(
        (CANVAS_DISPLAY_WIDTH, display_height)
    )

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

    col_save, col_apply, col_reset = st.columns(3)
    with col_save:
        if st.button(
            "保存警戒区",
            type="primary",
            use_container_width=True,
            disabled=new_polygon is None,
        ):
            zone.polygon_norm = new_polygon
            zone.expansion_px = float(expansion)
            if enabled:
                if _apply_camera_ground(
                    zone,
                    original_width,
                    original_height,
                    height_m,
                    pitch_deg,
                    hfov_deg,
                    roll_deg,
                ):
                    st.success(
                        f"已保存 {len(new_polygon)} 点地面 3D 区域，监控端将按脚底判定。"
                    )
            else:
                zone.disable_ground_3d()
                zone.save()
                st.success(f"已保存 {len(new_polygon)} 点 2D 多边形。")
    with col_apply:
        if st.button("应用地面标定", use_container_width=True):
            if enabled:
                if _apply_camera_ground(
                    zone,
                    original_width,
                    original_height,
                    height_m,
                    pitch_deg,
                    hfov_deg,
                    roll_deg,
                ):
                    st.success("已启用地面 3D。网格应对准地面接缝；不对就改高度/俯仰。")
                    st.rerun()
            else:
                zone.disable_ground_3d()
                zone.save()
                st.success("已切回 2D 多边形判定。")
                st.rerun()
    with col_reset:
        if st.button("恢复默认（左半屏）", use_container_width=True):
            zone.polygon_norm = default_polygon_norm()
            zone.expansion_px = float(expansion)
            if enabled:
                _apply_camera_ground(
                    zone,
                    original_width,
                    original_height,
                    height_m,
                    pitch_deg,
                    hfov_deg,
                    roll_deg,
                )
            else:
                zone.disable_ground_3d()
                zone.save()
            st.success("已恢复默认区域。")

    mode_label = {
        "camera": "地面 3D（相机几何）",
        "points": "地面 3D（4 点对应）",
        "off": "画面 2D",
    }.get(zone.ground_source, zone.ground_source)
    st.markdown(f"**当前判定：{mode_label}**")
    st.markdown("**当前已保存的归一化坐标：**")
    st.code(json.dumps(zone.polygon_norm, ensure_ascii=False, indent=2))

    with st.expander("进阶：4 组像素 ↔ 地面米对应点"):
        st.caption("棋盘格或卷尺量出的对应点。保存后会覆盖相机几何标定。")
        _homography_editor(zone, original_width, original_height)
