import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 自动扫描 CSV 文件 ==========
csv_files = glob.glob("*.csv")
if not csv_files:
    print("警告：当前目录没有找到 CSV 文件，将使用示例数据")
    csv_files = []

def load_csv(filepath):
    """读取 CSV 文件，返回第一列和第二列数据"""
    df = pd.read_csv(filepath)
    # 跳过表头，直接取数值列
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    y = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
    return x, y, df.columns.tolist()

# 如果有 CSV 文件，默认加载第一个
if csv_files:
    x_data, y_data, col_names = load_csv(csv_files[0])
else:
    # 备用示例数据
    x_data = np.arange(0, 21)
    y_data = np.array([0, 5, 1, 3, 2, 6, 4, 7, 3, 8, 2, 5, 9, 1, 6, 4, 7, 3, 8, 5, 2])
    col_names = ["x", "y"]

# ========== 创建图形 ==========
fig = plt.figure(figsize=(14, 10))

# ========== 3D 螺旋子图 ==========
ax_3d = fig.add_subplot(3, 1, 1, projection='3d')
t = np.linspace(0, 20 * np.pi, 200)
x_3d = np.cos(t)
y_3d = np.sin(t)
z_3d = t
ax_3d.plot(x_3d, y_3d, z_3d, color='blue', linewidth=2)
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.set_title("3D 螺旋线")

# ========== 2D 折线图子图 ==========
ax = fig.add_subplot(3, 1, 2)
fig.subplots_adjust(bottom=0.35)

# 滑窗参数
window_size = min(20, len(x_data))
start_idx = 0

# 初始窗口数据
x_window = x_data[start_idx:start_idx + window_size]
y_window = y_data[start_idx:start_idx + window_size]

line, = ax.plot(x_window, y_window, 'o-', linewidth=2, markersize=4)
ax.set_xlabel(col_names[0])
ax.set_ylabel(col_names[1])
ax.set_title(f"CSV 数据 - 滑窗显示 (窗口大小: {window_size})")

# 自动计算合理的坐标轴范围
def calc_axis_limits(data, padding=0.1):
    lo, hi = data.min(), data.max()
    pad = (hi - lo) * padding if hi > lo else 1.0
    return lo - pad, hi + pad

x_lo, x_hi = calc_axis_limits(x_data)
y_lo, y_hi = calc_axis_limits(y_data)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)

# ========== 控件区域布局 ==========
# CSV 文件选择器（右上角）
ax_radio = fig.add_axes([0.75, 0.65, 0.2, 0.15])
if csv_files:
    radio = RadioButtons(ax_radio, csv_files, active=0)
else:
    radio = None

# 滑窗控制滑块
ax_window = fig.add_axes([0.15, 0.25, 0.5, 0.03])
ax_start = fig.add_axes([0.15, 0.20, 0.5, 0.03])

# 坐标轴控制滑块
ax_y_max = fig.add_axes([0.15, 0.12, 0.5, 0.03])
ax_y_min = fig.add_axes([0.15, 0.07, 0.5, 0.03])
ax_x_min = fig.add_axes([0.15, 0.02, 0.5, 0.03])

# 创建滑块
slider_window = Slider(ax_window, '窗口大小', 5, len(x_data), valinit=window_size, valstep=1)
slider_start = Slider(ax_start, '起始位置', 0, len(x_data) - 5, valinit=0, valstep=1)

y_init_lo, y_init_hi = calc_axis_limits(y_window)
slider_y_min = Slider(ax_y_min, "Y 下限", y_lo, y_hi, valinit=y_init_lo, valstep=0.1)
slider_y_max = Slider(ax_y_max, "Y 上限", y_lo, y_hi, valinit=y_init_hi, valstep=0.1)

x_init_lo, x_init_hi = calc_axis_limits(x_window)
slider_x_min = Slider(ax_x_min, "X 下限", x_lo, x_hi, valinit=x_init_lo, valstep=0.1)

# ========== 回调函数 ==========
def update_window(_):
    """更新滑窗显示"""
    global window_size, start_idx
    window_size = int(slider_window.val)
    start_idx = int(slider_start.val)
    
    # 确保窗口不超出数据范围
    if start_idx + window_size > len(x_data):
        start_idx = len(x_data) - window_size
        slider_start.set_val(start_idx)
    
    x_win = x_data[start_idx:start_idx + window_size]
    y_win = y_data[start_idx:start_idx + window_size]
    
    line.set_xdata(x_win)
    line.set_ydata(y_win)
    
    # 自动调整坐标轴
    x_win_lo, x_win_hi = calc_axis_limits(x_win)
    y_win_lo, y_win_hi = calc_axis_limits(y_win)
    
    ax.set_xlim(x_win_lo, x_win_hi)
    ax.set_ylim(y_win_lo, y_win_hi)
    
    slider_x_min.valmin = x_lo
    slider_x_min.valmax = x_hi
    slider_x_min.set_val(x_win_lo)
    
    slider_y_min.valmin = y_lo
    slider_y_min.valmax = y_hi
    slider_y_min.set_val(y_win_lo)
    
    slider_y_max.valmin = y_lo
    slider_y_max.valmax = y_hi
    slider_y_max.set_val(y_win_hi)
    
    ax.set_title(f"CSV 数据 - 滑窗显示 (窗口: {window_size}, 起始: {start_idx})")
    fig.canvas.draw_idle()

def on_y_slider(_):
    lo, hi = sorted((slider_y_min.val, slider_y_max.val))
    if hi - lo < 1e-6:
        hi = lo + 0.01
    ax.set_ylim(lo, hi)
    fig.canvas.draw_idle()

def on_x_slider(_):
    xlo = slider_x_min.val
    xhi = x_data.max()
    if xhi - xlo < 1e-6:
        xhi = xlo + 0.01
    ax.set_xlim(xlo, xhi)
    fig.canvas.draw_idle()

def on_csv_selected(label):
    """切换 CSV 文件"""
    global x_data, y_data, col_names
    x_data, y_data, col_names = load_csv(label)
    
    # 重置滑块范围
    slider_window.valmax = len(x_data)
    slider_window.set_val(min(20, len(x_data)))
    
    slider_start.valmax = len(x_data) - 5
    slider_start.set_val(0)
    
    # 重新计算坐标轴范围
    global x_lo, x_hi, y_lo, y_hi
    x_lo, x_hi = calc_axis_limits(x_data)
    y_lo, y_hi = calc_axis_limits(y_data)
    
    update_window(None)
    ax.set_xlabel(col_names[0])
    ax.set_ylabel(col_names[1])
    fig.canvas.draw_idle()

# 绑定回调
slider_window.on_changed(update_window)
slider_start.on_changed(update_window)
slider_y_min.on_changed(on_y_slider)
slider_y_max.on_changed(on_y_slider)
slider_x_min.on_changed(on_x_slider)

if radio is not None:
    radio.on_clicked(on_csv_selected)

# 初始绘制
update_window(None)

plt.show()
