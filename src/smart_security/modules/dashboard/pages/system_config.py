# 模块：系统配置页
# 作用：管理截图清理，并把逗留阈值写回 config.yaml（非敏感配置）。

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from smart_security.core.config import DEFAULT_YAML_PATH, get_settings, reload_settings
from smart_security.modules.dashboard.screenshots import ScreenshotService
from smart_security.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CLEAN_DAYS = 7


def render(service: ScreenshotService) -> None:
    settings = get_settings()
    st.header("系统配置")

    st.subheader("报警设置")
    lingering_threshold = st.slider(
        "滞留报警阈值(秒)", 5, 60, int(settings.alarm.lingering_threshold)
    )
    save_screenshots = st.checkbox(
        "保存报警截图", value=settings.alarm.save_screenshots
    )
    alert_sound = st.checkbox("启用声音报警", value=settings.alarm.alert_sound)

    st.subheader("截图管理")
    st.info(f"截图目录: {service.screenshots_dir}")
    files = list(service.screenshots_dir.glob("*.jpg"))
    st.metric("截图数量", len(files))

    for index, file in enumerate(files[:10]):
        col_name, col_preview, col_delete = st.columns([3, 1, 1])
        col_name.write(f"{index + 1}. {file.name} ({file.stat().st_size // 1024} KB)")
        if col_preview.button("预览", key=f"prev_{index}", use_container_width=True):
            st.image(str(file), caption=file.name, use_column_width=True)
        if col_delete.button(
            "删除",
            key=f"cfg_del_{index}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.delete_confirmation = file.name
            st.rerun()
    if len(files) > 10:
        st.info(f"还有 {len(files) - 10} 个文件未显示")

    if st.session_state.delete_confirmation:
        filename = st.session_state.delete_confirmation
        st.warning(f"确认删除: {filename}")
        col_ok, col_cancel, _ = st.columns([1, 1, 2])
        if col_ok.button(
            "确认", key="cfg_confirm", type="primary", use_container_width=True
        ):
            try:
                service.delete_file(str(service.screenshots_dir / filename))
                st.success(f"已删除: {filename}")
            except FileNotFoundError:
                st.error(f"文件不存在: {filename}")
            st.session_state.delete_confirmation = None
            st.rerun()
        if col_cancel.button("取消", key="cfg_cancel", use_container_width=True):
            st.session_state.delete_confirmation = None
            st.rerun()

    st.subheader("批量清理")
    col_old, col_all = st.columns(2)
    with col_old:
        st.write(f"**清理 {DEFAULT_CLEAN_DAYS} 天前的截图**")
        if st.button("执行清理", key="clean_old", use_container_width=True):
            deleted = service.clean_old(days=DEFAULT_CLEAN_DAYS)
            if deleted:
                st.success(f"已删除 {deleted} 个旧截图")
                st.rerun()
            else:
                st.info(f"无 {DEFAULT_CLEAN_DAYS} 天前的截图")
    with col_all:
        st.write("**清理全部截图**")
        if st.session_state.delete_all_confirmation:
            st.warning("此操作不可恢复！")
            col_confirm, col_abort = st.columns(2)
            if col_confirm.button(
                "确认删除全部",
                key="confirm_all",
                type="primary",
                use_container_width=True,
            ):
                deleted = service.clean_all()
                st.session_state.delete_all_confirmation = False
                st.success(f"已删除 {deleted} 个截图")
                st.rerun()
            if col_abort.button("取消", key="cancel_all", use_container_width=True):
                st.session_state.delete_all_confirmation = False
                st.rerun()
        elif st.button(
            "执行清理", key="clean_all", use_container_width=True, type="secondary"
        ):
            st.session_state.delete_all_confirmation = True
            st.rerun()

    if st.button("保存配置", use_container_width=True):
        _persist_alarm_settings(lingering_threshold, save_screenshots, alert_sound)
        reload_settings()
        st.success("配置已保存到 config.yaml")


def _persist_alarm_settings(
    lingering_threshold: int, save_screenshots: bool, alert_sound: bool
) -> None:
    path: Path = DEFAULT_YAML_PATH
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    alarm = data.setdefault("alarm", {})
    alarm["lingering_threshold"] = lingering_threshold
    alarm["save_screenshots"] = save_screenshots
    alarm["alert_sound"] = alert_sound
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    logger.info("Updated alarm settings in %s", path)
