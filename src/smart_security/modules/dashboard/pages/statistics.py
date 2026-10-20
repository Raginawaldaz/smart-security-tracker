# 模块：报警统计页
# 作用：按日期筛选截图记录并绘制分布图。

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from smart_security.modules.dashboard.screenshots import ScreenshotService


def render(service: ScreenshotService) -> None:
    st.header("报警统计分析")
    alerts = service.list_alerts()
    if not alerts:
        st.info("暂无报警记录，请先运行监控系统")
        return

    frame = pd.DataFrame(alerts)
    frame["date"] = frame["timestamp"].dt.date
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.day_name()

    st.subheader("筛选条件")
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("开始日期", value=frame["date"].min())
    end_date = col_end.date_input("结束日期", value=frame["date"].max())

    filtered = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
    if filtered.empty:
        st.warning("选定日期范围内无报警记录")
        return

    days_count = max((end_date - start_date).days + 1, 1)
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("报警总数", len(filtered))
    metric2.metric("涉及目标数", filtered["track_id"].nunique())
    metric3.metric("日均报警", f"{len(filtered) / days_count:.1f}")
    metric4.metric("最常报警目标", filtered["track_id"].mode()[0])

    col_daily, col_hourly = st.columns(2)
    with col_daily:
        date_counts = filtered.groupby("date").size().reset_index(name="count")
        st.plotly_chart(
            px.bar(date_counts, x="date", y="count", title="每日报警次数"),
            use_container_width=True,
        )
    with col_hourly:
        hour_counts = filtered.groupby("hour").size().reset_index(name="count")
        st.plotly_chart(
            px.bar(hour_counts, x="hour", y="count", title="每小时报警次数"),
            use_container_width=True,
        )

    st.subheader("目标分析")
    track_counts = (
        filtered.groupby("track_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    col_table, col_pie = st.columns(2)
    col_table.dataframe(track_counts, use_container_width=True)
    col_pie.plotly_chart(
        px.pie(
            track_counts.head(10),
            values="count",
            names="track_id",
            title="目标报警分布（前10名）",
        ),
        use_container_width=True,
    )
