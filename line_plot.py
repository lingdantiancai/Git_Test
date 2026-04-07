import glob
import numpy as np
import pandas as pd
import matplotlib as mpl

# 配置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# ========== 自动扫描 CSV 文件 ==========
csv_files = glob.glob("*.csv")
if not csv_files:
    print("错误：当前目录没有找到 CSV 文件")
    exit(1)

def load_csv(filepath):
    """读取 CSV 文件，返回时间和两列数据"""
    df = pd.read_csv(filepath, header=1)  # 跳过两行表头
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    col1 = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
    col2 = pd.to_numeric(df.iloc[:, 2], errors='coerce').dropna().values
    col_names = df.columns[:3].tolist()
    return time, col1, col2, col_names

# 加载第一个 CSV
time_data, disp_data, force_data, col_names = load_csv(csv_files[0])
total_points = len(time_data)

# 固定窗口宽度
WINDOW_SIZE = 20

# ========== 创建图形 ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
fig.subplots_adjust(bottom=0.25, hspace=0.35)

# 初始空数据
line1, = ax1.plot([], [], 'b-', linewidth=1.5, marker='o', markersize=3)
line2, = ax2.plot([], [], 'r-', linewidth=1.5, marker='o', markersize=3)

ax1.set_ylabel(col_names[1], fontsize=12)
ax1.set_title(f"动态加载: {csv_files[0]} (窗口大小: {WINDOW_SIZE})", fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.set_ylabel(col_names[2], fontsize=12)
ax2.set_xlabel(col_names[0], fontsize=12)
ax2.grid(True, alpha=0.3)

# ========== 动画控制变量 ==========
current_frame = [0]  # 当前帧（已加载的数据点数）
is_playing = [True]

# ========== 速度滑块 ==========
ax_speed = fig.add_axes([0.15, 0.10, 0.5, 0.03])
slider_speed = Slider(ax_speed, '速度 (帧/秒)', 1, 100, valinit=30, valstep=1)

# ========== 播放/暂停按钮 ==========
ax_pause = fig.add_axes([0.70, 0.09, 0.1, 0.04])
btn_pause = Button(ax_pause, '暂停')

def update(frame):
    """动画更新函数"""
    if not is_playing[0]:
        return line1, line2
    
    current_frame[0] = frame
    
    # 当前窗口：从 0 到 frame，但最多显示 WINDOW_SIZE 个点
    end_idx = min(frame, total_points)
    start_idx = max(0, end_idx - WINDOW_SIZE)
    
    # 更新位移图
    x1 = time_data[start_idx:end_idx]
    y1 = disp_data[start_idx:end_idx]
    line1.set_data(x1, y1)
    
    # 更新力图
    y2 = force_data[start_idx:end_idx]
    line2.set_data(x1, y2)
    
    # 自动调整坐标轴
    if len(x1) > 0:
        ax1.set_xlim(x1.min(), x1.max())
        ax2.set_xlim(x1.min(), x1.max())
        
        y1_lo, y1_hi = y1.min(), y1.max()
        pad1 = (y1_hi - y1_lo) * 0.1 if y1_hi > y1_lo else 0.1
        ax1.set_ylim(y1_lo - pad1, y1_hi + pad1)
        
        y2_lo, y2_hi = y2.min(), y2.max()
        pad2 = (y2_hi - y2_lo) * 0.1 if y2_hi > y2_lo else 0.1
        ax2.set_ylim(y2_lo - pad2, y2_hi + pad2)
    
    ax1.set_title(f"动态加载: {csv_files[0]} - 已加载: {end_idx}/{total_points} 点", fontsize=14)
    
    return line1, line2

def on_pause_clicked(event):
    """暂停/播放按钮回调"""
    is_playing[0] = not is_playing[0]
    btn_pause.label.set_text('继续' if not is_playing[0] else '暂停')
    fig.canvas.draw_idle()

btn_pause.on_clicked(on_pause_clicked)

# 创建动画
def get_interval():
    return 1000.0 / slider_speed.val

ani = FuncAnimation(
    fig, update,
    frames=range(1, total_points + 1),
    interval=get_interval(),
    blit=False,
    repeat=False
)

# 速度变化时更新间隔
def on_speed_change(val):
    ani.event_source.interval = get_interval()

slider_speed.on_changed(on_speed_change)

plt.show()
