# 模块：Streamlit 仪表板入口
# 作用：配置页面样式与侧边栏路由，具体页面逻辑下放到 pages/。

from __future__ import annotations

import streamlit as st

from smart_security.modules.dashboard.pages import (
    overview,
    statistics,
    system_config,
    zone_editor,
)
from smart_security.modules.dashboard.screenshots import ScreenshotService

st.set_page_config(page_title="智能安防监控系统", page_icon="S", layout="wide")

st.markdown(
    """
<style>
    .header      { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: .5rem;
                   margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
    .alert-card  { background-color: #ffebee; border-left: 4px solid #f44336;
                   padding: 1rem; border-radius: .5rem; margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

PAGE_OVERVIEW = "仪表板"
PAGE_STATS = "统计分析"
PAGE_ZONE = "警戒区配置"
PAGE_SYSTEM = "系统配置"


def run() -> None:
    if "delete_confirmation" not in st.session_state:
        st.session_state.delete_confirmation = None
    if "delete_all_confirmation" not in st.session_state:
        st.session_state.delete_all_confirmation = False

    service = ScreenshotService()
    st.sidebar.title("导航菜单")
    page = st.sidebar.selectbox(
        "选择页面", [PAGE_OVERVIEW, PAGE_STATS, PAGE_ZONE, PAGE_SYSTEM]
    )

    st.sidebar.subheader("系统状态")
    files = list(service.screenshots_dir.glob("*.jpg"))
    st.sidebar.write(f"截图文件数: {len(files)}")
    if files:
        total_mb = sum(file.stat().st_size for file in files) / 1_048_576
        st.sidebar.write(f"截图总大小: {total_mb:.2f} MB")

    if page == PAGE_OVERVIEW:
        overview.render(service)
    elif page == PAGE_STATS:
        statistics.render(service)
    elif page == PAGE_ZONE:
        zone_editor.render()
    else:
        system_config.render(service)


if __name__ == "__main__":
    run()
