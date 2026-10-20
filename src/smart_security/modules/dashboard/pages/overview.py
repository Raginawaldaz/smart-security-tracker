# 模块：仪表板总览页
# 作用：展示报警计数与最近截图，删除确认走 session_state 避免误删。

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import streamlit as st

from smart_security.modules.dashboard.screenshots import ScreenshotService


def render_alert_row(
    service: ScreenshotService, alert: dict[str, Any], index: int, prefix: str = ""
) -> None:
    if st.session_state.delete_confirmation == alert["filename"]:
        st.warning(f"确认删除: {alert['filename']}")
        col_ok, col_cancel, _ = st.columns([1, 1, 2])
        with col_ok:
            if st.button(
                "确认删除",
                key=f"{prefix}_confirm_{index}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    service.delete_file(alert["filepath"])
                    st.success(f"已删除: {alert['filename']}")
                except FileNotFoundError:
                    st.error(f"文件不存在: {alert['filename']}")
                st.session_state.delete_confirmation = None
                st.rerun()
        with col_cancel:
            if st.button(
                "取消", key=f"{prefix}_cancel_{index}", use_container_width=True
            ):
                st.session_state.delete_confirmation = None
                st.rerun()
        st.divider()
        return

    st.markdown(f"**报警记录 #{index + 1}**")
    col_img, col_info = st.columns([1, 2])
    with col_img:
        try:
            st.image(
                alert["filepath"],
                use_column_width=True,
                caption=(
                    f"ID: {alert['track_id']} — "
                    f"{alert['timestamp'].strftime('%H:%M:%S')}"
                ),
            )
        except (OSError, ValueError) as exc:
            st.error(f"无法加载截图: {exc}")
    with col_info:
        st.write(f"**时间:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**目标 ID:** {alert['track_id']}")
        st.write(f"**文件名:** {alert['filename']}")
        col_dl, col_del = st.columns(2)
        with col_dl:
            try:
                with open(alert["filepath"], "rb") as handle:
                    st.download_button(
                        "下载截图",
                        handle.read(),
                        file_name=alert["filename"],
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"{prefix}_dl_{index}",
                    )
            except OSError as exc:
                st.error(f"下载失败: {exc}")
        with col_del:
            if st.button(
                "删除截图",
                type="secondary",
                use_container_width=True,
                key=f"{prefix}_del_{index}",
            ):
                st.session_state.delete_confirmation = alert["filename"]
                st.rerun()
    st.divider()


def render(service: ScreenshotService) -> None:
    st.markdown(
        '<h1 class="header">智能安防监控系统</h1>',
        unsafe_allow_html=True,
    )
    alerts = service.list_alerts()
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    today_alerts = [item for item in alerts if item["timestamp"].date() == today]
    weekly_alerts = [item for item in alerts if item["timestamp"].date() >= week_ago]

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "总报警次数", len(alerts)),
        (col2, "今日报警", len(today_alerts)),
        (col3, "本周报警", len(weekly_alerts)),
        (col4, "涉及目标", len({item["track_id"] for item in alerts})),
    ]
    for column, label, value in metrics:
        with column:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("最近报警记录")
    if not alerts:
        st.info("暂无报警记录")
        return

    latest = sorted(alerts, key=lambda item: item["timestamp"], reverse=True)[:5]
    for index, alert in enumerate(latest):
        render_alert_row(service, alert, index, prefix="dash")
