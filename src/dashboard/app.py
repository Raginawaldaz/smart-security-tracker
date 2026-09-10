"""
Streamlit 安防监控仪表板
从项目根目录运行: streamlit run src/dashboard/app.py
"""
import streamlit as st
from pathlib import Path
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(
    page_title="智能安防监控系统",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
<style>
    .header      { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: .5rem;
                   margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
    .alert-card  { background-color: #ffebee; border-left: 4px solid #f44336;
                   padding: 1rem; border-radius: .5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# 数据目录始终指向项目根目录下的 data/
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class SecurityDashboard:
    def __init__(self):
        self.data_dir = _PROJECT_ROOT / "data"
        self.screenshots_dir = self.data_dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        if 'delete_confirmation' not in st.session_state:
            st.session_state.delete_confirmation = None
        if 'delete_all_confirmation' not in st.session_state:
            st.session_state.delete_all_confirmation = False

    # ------------------------------------------------------------------ data

    def get_alert_stats(self):
        """从截图文件名解析报警记录"""
        try:
            files = list(self.screenshots_dir.glob("*.jpg"))
            st.sidebar.success(f"找到 {len(files)} 个截图文件")

            alerts = []
            patterns = [
                r"lingering_(\d+)_(\d{8}_\d{6})\.jpg",
                r"lingering_(\d+)_(\d+)\.jpg",
                r"(\d+)_(\d{8}_\d{6})\.jpg",
            ]

            for file in files:
                matched = False
                for pat in patterns:
                    m = re.match(pat, file.name)
                    if m:
                        track_id = int(m.group(1))
                        ts_str = m.group(2)
                        try:
                            ts = (datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                                  if '_' in ts_str
                                  else datetime.fromtimestamp(int(ts_str)))
                        except Exception:
                            ts = datetime.fromtimestamp(os.path.getmtime(file))
                        alerts.append({
                            "timestamp": ts, "track_id": track_id,
                            "filepath": str(file), "filename": file.name,
                        })
                        matched = True
                        break

                if not matched:
                    st.sidebar.warning(f"无法解析文件名: {file.name}")
                    alerts.append({
                        "timestamp": datetime.fromtimestamp(os.path.getmtime(file)),
                        "track_id": 0, "filepath": str(file), "filename": file.name,
                    })

            st.sidebar.success(f"成功解析 {len(alerts)} 条记录")
            return alerts
        except Exception as e:
            st.sidebar.error(f"获取报警统计时出错: {e}")
            return []

    # ------------------------------------------------------------------ file ops

    def delete_single_screenshot(self, file_path, filename):
        p = Path(file_path)
        if p.exists():
            p.unlink()
            st.success(f"已删除: {filename}")
            st.session_state.delete_confirmation = None
            st.rerun()
        else:
            st.error(f"文件不存在: {filename}")

    def clean_old_screenshots(self, days=7):
        cutoff = datetime.now() - timedelta(days=days)
        count = 0
        for f in self.screenshots_dir.glob("*.jpg"):
            if datetime.fromtimestamp(os.path.getmtime(f)) < cutoff:
                f.unlink()
                count += 1
        return count

    def clean_all_screenshots(self):
        files = list(self.screenshots_dir.glob("*.jpg"))
        for f in files:
            f.unlink()
        return len(files)

    # ------------------------------------------------------------------ pages

    def show_dashboard(self):
        st.markdown('<h1 class="header">🎥 智能安防监控系统</h1>', unsafe_allow_html=True)
        alerts = self.get_alert_stats()
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)

        today_alerts  = [a for a in alerts if a["timestamp"].date() == today]
        weekly_alerts = [a for a in alerts if a["timestamp"].date() >= week_ago]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in [
            (c1, "📊 总报警次数", len(alerts)),
            (c2, "📅 今日报警",   len(today_alerts)),
            (c3, "📈 本周报警",   len(weekly_alerts)),
            (c4, "🎯 涉及目标",   len({a["track_id"] for a in alerts})),
        ]:
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, value)
                st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🆕 最近报警记录")
        if not alerts:
            st.info("暂无报警记录")
            return

        for i, alert in enumerate(sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:5]):
            self._render_alert_row(alert, i, prefix="dash")

    def _render_alert_row(self, alert, i, prefix=""):
        """可复用的单条报警渲染"""
        if st.session_state.delete_confirmation == alert["filename"]:
            st.warning(f"⚠️ 确认删除: {alert['filename']}")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ 确认删除", key=f"{prefix}_confirm_{i}",
                             type="primary", use_container_width=True):
                    self.delete_single_screenshot(alert["filepath"], alert["filename"])
            with c2:
                if st.button("❌ 取消", key=f"{prefix}_cancel_{i}",
                             use_container_width=True):
                    st.session_state.delete_confirmation = None
                    st.rerun()
            st.divider()
            return

        st.markdown(f"**报警记录 #{i + 1}**")
        col_img, col_info = st.columns([1, 2])
        with col_img:
            try:
                st.image(alert["filepath"], use_column_width=True,
                         caption=f"ID: {alert['track_id']} — {alert['timestamp'].strftime('%H:%M:%S')}")
            except Exception as e:
                st.error(f"无法加载截图: {e}")
        with col_info:
            st.write(f"**时间:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**目标 ID:** {alert['track_id']}")
            st.write(f"**文件名:** {alert['filename']}")
            cb1, cb2 = st.columns(2)
            with cb1:
                try:
                    with open(alert["filepath"], "rb") as f:
                        st.download_button("📥 下载截图", f.read(),
                                           file_name=alert["filename"], mime="image/jpeg",
                                           use_container_width=True, key=f"{prefix}_dl_{i}")
                except Exception as e:
                    st.error(f"下载失败: {e}")
            with cb2:
                if st.button("🗑️ 删除截图", type="secondary",
                             use_container_width=True, key=f"{prefix}_del_{i}"):
                    st.session_state.delete_confirmation = alert["filename"]
                    st.rerun()
        st.divider()

    def show_statistics(self):
        st.header("📊 报警统计分析")
        alerts = self.get_alert_stats()
        if not alerts:
            st.info("暂无报警记录，请先运行监控系统")
            return

        df = pd.DataFrame(alerts)
        df['date']        = df['timestamp'].dt.date
        df['hour']        = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

        st.subheader("📅 筛选条件")
        c1, c2 = st.columns(2)
        start_date = c1.date_input("开始日期", value=df['date'].min())
        end_date   = c2.date_input("结束日期", value=df['date'].max())

        fdf = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if fdf.empty:
            st.warning("选定日期范围内无报警记录")
            return

        days_count = max((end_date - start_date).days + 1, 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("报警总数",    len(fdf))
        m2.metric("涉及目标数",  fdf['track_id'].nunique())
        m3.metric("日均报警",    f"{len(fdf) / days_count:.1f}")
        m4.metric("最常报警目标", fdf['track_id'].mode()[0])

        c1, c2 = st.columns(2)
        with c1:
            date_counts = fdf.groupby('date').size().reset_index(name='count')
            st.plotly_chart(px.bar(date_counts, x='date', y='count', title='每日报警次数'),
                            use_container_width=True)
        with c2:
            hour_counts = fdf.groupby('hour').size().reset_index(name='count')
            st.plotly_chart(px.bar(hour_counts, x='hour', y='count', title='每小时报警次数'),
                            use_container_width=True)

        st.subheader("🎯 目标分析")
        tc = fdf.groupby('track_id').size().reset_index(name='count').sort_values('count', ascending=False)
        c1, c2 = st.columns(2)
        c1.dataframe(tc, use_container_width=True)
        c2.plotly_chart(px.pie(tc.head(10), values='count', names='track_id',
                               title='目标报警分布（前10名）'), use_container_width=True)

    def show_configuration(self):
        st.header("⚙️ 系统配置")

        st.subheader("报警设置")
        st.slider("滞留报警阈值(秒)", 5, 60, 15)
        st.checkbox("保存报警截图", value=True)
        st.checkbox("启用声音报警", value=True)

        st.subheader("截图管理")
        st.info(f"截图目录: {self.screenshots_dir}")
        files = list(self.screenshots_dir.glob("*.jpg"))
        st.metric("截图数量", len(files))

        for i, f in enumerate(files[:10]):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"{i+1}. {f.name} ({f.stat().st_size // 1024} KB)")
            if c2.button("👁️ 预览", key=f"prev_{i}", use_container_width=True):
                st.image(str(f), caption=f.name, use_column_width=True)
            if c3.button("🗑️ 删除", key=f"cfg_del_{i}", use_container_width=True, type="secondary"):
                st.session_state.delete_confirmation = f.name
                st.rerun()
        if len(files) > 10:
            st.info(f"还有 {len(files) - 10} 个文件未显示")

        if st.session_state.delete_confirmation:
            fn = st.session_state.delete_confirmation
            st.warning(f"⚠️ 确认删除: {fn}")
            c1, c2, _ = st.columns([1, 1, 2])
            if c1.button("✅ 确认", key="cfg_confirm", type="primary", use_container_width=True):
                self.delete_single_screenshot(str(self.screenshots_dir / fn), fn)
            if c2.button("❌ 取消", key="cfg_cancel", use_container_width=True):
                st.session_state.delete_confirmation = None
                st.rerun()

        st.subheader("批量清理")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**清理 7 天前的截图**")
            if st.button("执行清理", key="clean_old", use_container_width=True):
                n = self.clean_old_screenshots(days=7)
                st.success(f"已删除 {n} 个旧截图") if n else st.info("无 7 天前的截图")
                if n:
                    st.rerun()
        with c2:
            st.write("**清理全部截图**")
            if st.session_state.delete_all_confirmation:
                st.warning("此操作不可恢复！")
                ca, cb = st.columns(2)
                if ca.button("✅ 确认删除全部", key="confirm_all",
                             type="primary", use_container_width=True):
                    n = self.clean_all_screenshots()
                    st.session_state.delete_all_confirmation = False
                    st.success(f"已删除 {n} 个截图")
                    st.rerun()
                if cb.button("❌ 取消", key="cancel_all", use_container_width=True):
                    st.session_state.delete_all_confirmation = False
                    st.rerun()
            else:
                if st.button("执行清理", key="clean_all",
                             use_container_width=True, type="secondary"):
                    st.session_state.delete_all_confirmation = True
                    st.rerun()

        if st.button("保存配置", use_container_width=True):
            st.success("配置已保存！")

    # ------------------------------------------------------------------ zone editor

    def _load_zone(self):
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from src.core.zone_config import ZoneConfig
        return ZoneConfig()

    @staticmethod
    def _parse_canvas_polygon(objects, disp_w, disp_h):
        """从 drawable-canvas 返回对象解析出归一化多边形点（支持矩形/多边形）。"""
        if not objects:
            return None
        obj = objects[-1]  # 取最后绘制的形状
        pts = []
        otype = obj.get("type")
        if otype == "rect":
            left = obj.get("left", 0); top = obj.get("top", 0)
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            pts = [[left, top], [left + w, top], [left + w, top + h], [left, top + h]]
        elif otype in ("path", "polygon"):
            raw = obj.get("path") or obj.get("points")
            if raw:
                for seg in raw:
                    if isinstance(seg, dict) and "x" in seg:
                        pts.append([seg["x"], seg["y"]])
                    elif isinstance(seg, (list, tuple)):
                        nums = [v for v in seg if isinstance(v, (int, float))]
                        if len(nums) >= 2:
                            pts.append([nums[-2], nums[-1]])
        if len(pts) < 3:
            return None
        return [[max(0.0, min(1.0, x / disp_w)), max(0.0, min(1.0, y / disp_h))]
                for x, y in pts]

    def show_zone_config(self):
        st.header("🛡️ 警戒区配置（可拖拽）")
        st.caption("在画面上拖拽绘制覆盖范围；保存后监控端会自动热加载，无需重启。")

        try:
            from streamlit_drawable_canvas import st_canvas
            from PIL import Image
        except Exception:
            st.error("缺少依赖：请先安装 `pip install streamlit-drawable-canvas pillow`")
            return

        zone = self._load_zone()
        frame_path = self.data_dir / "last_frame.jpg"

        cc1, cc2 = st.columns([3, 1])
        with cc2:
            if st.button("📷 从摄像头抓拍底图", use_container_width=True):
                self._capture_frame(frame_path)
            mode = st.radio("绘制方式", ["polygon", "rect"],
                            help="polygon 可贴合透视地面；rect 为快速矩形")
            expansion = st.slider("边界外扩(px)", 0, 60, int(zone.expansion_px))

        if not frame_path.exists():
            with cc1:
                st.info("暂无底图。请先运行监控端生成 data/last_frame.jpg，或点击右侧抓拍。")
            return

        bg = Image.open(str(frame_path)).convert("RGB")
        ow, oh = bg.size
        disp_w = 760
        disp_h = int(oh * disp_w / ow)
        bg_disp = bg.resize((disp_w, disp_h))

        with cc1:
            canvas = st_canvas(
                fill_color="rgba(0, 255, 0, 0.25)",
                stroke_color="#00FF00",
                stroke_width=2,
                background_image=bg_disp,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode=mode,
                key="zone_canvas",
            )

        objects = (canvas.json_data or {}).get("objects", []) if canvas else []
        new_poly = self._parse_canvas_polygon(objects, disp_w, disp_h)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 保存警戒区", type="primary", use_container_width=True,
                         disabled=new_poly is None):
                zone.polygon_norm = new_poly
                zone.expansion_px = float(expansion)
                zone.save()
                st.success(f"已保存 {len(new_poly)} 点多边形，监控端将自动生效。")
        with c2:
            if st.button("↩️ 恢复默认（左半屏）", use_container_width=True):
                from src.core.zone_config import default_polygon_norm
                zone.polygon_norm = default_polygon_norm()
                zone.expansion_px = float(expansion)
                zone.save()
                st.success("已恢复默认区域。")

        st.markdown("**当前已保存的归一化坐标：**")
        st.code(json.dumps(zone.polygon_norm, ensure_ascii=False, indent=2))

        with st.expander("🧭 地面投影（3D）标定 — 进阶可选"):
            st.caption("输入 4 组『图像像素 → 地面坐标(米)』对应点，用于把脚的位置投影到地面平面判断，"
                       "更适合高位俯视相机。留空则使用 2D 多边形判定。")
            self._homography_editor(zone, ow, oh)

    def _homography_editor(self, zone, ow, oh):
        import cv2
        st.write("图像点以原始分辨率像素填写（当前底图：{}×{}）。".format(ow, oh))
        img_pts, gnd_pts = [], []
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                st.markdown(f"**点 {i+1}**")
                ix = st.number_input(f"img_x{i}", value=0.0, key=f"ix{i}")
                iy = st.number_input(f"img_y{i}", value=0.0, key=f"iy{i}")
                gx = st.number_input(f"gnd_x{i}", value=0.0, key=f"gx{i}")
                gy = st.number_input(f"gnd_y{i}", value=0.0, key=f"gy{i}")
                img_pts.append([ix, iy]); gnd_pts.append([gx, gy])

        ca, cb = st.columns(2)
        with ca:
            if st.button("🧮 计算并启用地面投影", use_container_width=True):
                src = np.array(img_pts, dtype=np.float64)
                dst = np.array(gnd_pts, dtype=np.float64)
                if np.allclose(src, 0) or np.allclose(dst, 0):
                    st.warning("请填入有效的 4 组对应点。")
                else:
                    H, _ = cv2.findHomography(src, dst)
                    if H is None:
                        st.error("单应矩阵求解失败，请检查点位（勿共线）。")
                    else:
                        zone.homography = H
                        # 地面多边形：把当前图像多边形顶点投影到地面坐标
                        gp = []
                        for nx, ny in zone.polygon_norm:
                            v = H @ np.array([nx * ow, ny * oh, 1.0])
                            if abs(v[2]) > 1e-9:
                                gp.append([float(v[0] / v[2]), float(v[1] / v[2])])
                        zone.ground_polygon = np.array(gp) if len(gp) >= 3 else None
                        zone.save()
                        st.success("已启用地面投影（3D）判定。")
        with cb:
            if st.button("🚫 关闭地面投影（回到 2D）", use_container_width=True):
                zone.homography = None
                zone.ground_polygon = None
                zone.save()
                st.success("已切回 2D 多边形判定。")

    def _capture_frame(self, frame_path):
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(frame_path), frame)
            st.success("已抓拍底图。")
            st.rerun()
        else:
            st.error("无法从摄像头抓拍，请确认摄像头未被监控端占用。")

    # ------------------------------------------------------------------ entry

    def run(self):
        st.sidebar.title("导航菜单")
        page = st.sidebar.selectbox("选择页面", ["仪表板", "统计分析", "警戒区配置", "系统配置"])

        st.sidebar.subheader("系统状态")
        files = list(self.screenshots_dir.glob("*.jpg"))
        st.sidebar.write(f"截图文件数: {len(files)}")
        if files:
            total_mb = sum(f.stat().st_size for f in files) / 1_048_576
            st.sidebar.write(f"截图总大小: {total_mb:.2f} MB")

        {"仪表板": self.show_dashboard,
         "统计分析": self.show_statistics,
         "警戒区配置": self.show_zone_config,
         "系统配置": self.show_configuration}[page]()


if __name__ == "__main__":
    SecurityDashboard().run()
