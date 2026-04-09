"""CSV 力-位移数据动态可视化，集成信号处理管线。"""

from __future__ import annotations

import glob
import sys

import matplotlib as mpl

# 配置中文字体（需在 import pyplot 之前设置）
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

from agent_b_signal import process_signal

# ---------- 配置 ----------
DEFAULT_SPEED = 30  # 点/秒

# 信号处理配置
SIGNAL_PROCESSING_CONFIG = {
    "lowpass_cutoff_ratio": 0.05,  # fs * 0.05 = fs/20
    "outlier_method": "iqr",      # "iqr", "zscore", "hampel"
    "skip_outliers": False,
    "skip_lowpass": False,
}


def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """读取 CSV 文件，返回 (time, displacement, force, 列名列表)。

    CSV 格式预期：第一行为列名（如 Time,Displacement,Force），
    第二行为单位行（如 (s),(mm),(N)），从第三行起为数据。
    """
    df = pd.read_csv(filepath, header=0, skiprows=[1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    time = df.iloc[:, 0].values
    col1 = df.iloc[:, 1].values
    col2 = df.iloc[:, 2].values
    col_names = df.columns[:3].tolist()
    return time, col1, col2, col_names


def apply_signal_processing(
    disp: np.ndarray, force: np.ndarray, time: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """对位移和力数据应用信号处理管线。"""
    disp_series = pd.Series(disp, name="Displacement")
    force_series = pd.Series(force, name="Force")

    # 使用配置参数
    lowpass_cutoff = fs * SIGNAL_PROCESSING_CONFIG["lowpass_cutoff_ratio"]
    outlier_method = SIGNAL_PROCESSING_CONFIG["outlier_method"]
    skip_outliers = SIGNAL_PROCESSING_CONFIG["skip_outliers"]
    skip_lowpass = SIGNAL_PROCESSING_CONFIG["skip_lowpass"]

    disp_processed = process_signal(
        disp_series,
        fs=fs,
        lowpass_cutoff=lowpass_cutoff,
        outlier_method=outlier_method,
        skip_outliers=skip_outliers,
        skip_lowpass=skip_lowpass,
    ).values

    force_processed = process_signal(
        force_series,
        fs=fs,
        lowpass_cutoff=lowpass_cutoff,
        outlier_method=outlier_method,
        skip_outliers=skip_outliers,
        skip_lowpass=skip_lowpass,
    ).values

    return disp_processed, force_processed


class ForceDisplacementViewer:
    """力-位移曲线动态可视化器。"""

    def __init__(
        self,
        disp: np.ndarray,
        force: np.ndarray,
        col_names: list[str],
        filename: str,
    ) -> None:
        self.disp = disp
        self.force = force
        self.col_names = col_names
        self.filename = filename
        self.total_points = len(disp)
        self.current_idx = 0
        self.is_playing = False
        self.speed = DEFAULT_SPEED  # 点/秒
        self.accumulated_error = 0.0  # 累积误差，用于精确速度控制

        # 固定坐标轴范围（基于完整数据）
        self.x_min, self.x_max = disp.min(), disp.max()
        self.y_min, self.y_max = force.min(), force.max()

        self._build_ui()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)

        (self.line,) = self.ax.plot([], [], "b-", linewidth=1.5)

        self.ax.set_xlabel(f"{self.col_names[1]} (位移)", fontsize=13)
        self.ax.set_ylabel(f"{self.col_names[2]} (力)", fontsize=13)
        self.ax.set_title(f"力-位移曲线: {self.filename}", fontsize=14)
        self.ax.grid(True, alpha=0.3)

        # 固定坐标轴范围
        pad_x = (self.x_max - self.x_min) * 0.05 if self.x_max > self.x_min else 0.5
        pad_y = (self.y_max - self.y_min) * 0.05 if self.y_max > self.y_min else 0.5
        self.ax.set_xlim(self.x_min - pad_x, self.x_max + pad_x)
        self.ax.set_ylim(self.y_min - pad_y, self.y_max + pad_y)

        # 速度滑块：每秒读取的点数
        ax_speed = self.fig.add_axes([0.15, 0.10, 0.5, 0.03])
        self.slider_speed = Slider(
            ax_speed, "速度 (点/秒)", 1, 500, valinit=DEFAULT_SPEED, valstep=1
        )
        self.slider_speed.on_changed(self._on_speed_changed)

        # 播放/暂停按钮
        ax_pause = self.fig.add_axes([0.70, 0.09, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, "播放")
        self.btn_pause.on_clicked(self._on_pause_clicked)

    # ---- 回调 ----

    def _on_pause_clicked(self, _event) -> None:
        if not self.is_playing and self.current_idx >= self.total_points:
            # 播放完毕后点击，从头开始
            self.current_idx = 0
            self.accumulated_error = 0.0  # 重置累积误差
        self.is_playing = not self.is_playing
        self.btn_pause.label.set_text("暂停" if self.is_playing else "继续")
        self.fig.canvas.draw_idle()

    def _on_speed_changed(self, val: float) -> None:
        self.speed = int(val)

    # ---- 动画 ----

    def _tick(self) -> None:
        """定时器回调：每帧推进若干个数据点。"""
        if not self.is_playing or self.current_idx >= self.total_points:
            if self.current_idx >= self.total_points:
                self.is_playing = False
                self.btn_pause.label.set_text("重播")
                self.fig.canvas.draw_idle()
            return

        # 精确速度控制：使用累积误差补偿
        # 每帧理论推进点数 = 速度(点/秒) / 定时器频率(帧/秒)
        target_increment = self.speed / self.timer_interval_hz

        # 加上累积误差
        total_increment = target_increment + self.accumulated_error

        # 整数部分作为实际推进步数，小数部分作为新的累积误差
        steps = int(total_increment)
        self.accumulated_error = total_increment - steps

        # 至少推进1点（除非速度为0，但速度滑块最小为1）
        if steps < 1 and self.speed > 0:
            # 如果累积误差足够大，推进1点并减少累积误差
            if self.accumulated_error >= 1.0:
                steps = 1
                self.accumulated_error -= 1.0
            else:
                steps = 1  # 保证至少推进1点

        self.current_idx = min(self.current_idx + steps, self.total_points)

        x = self.disp[: self.current_idx]
        y = self.force[: self.current_idx]
        self.line.set_data(x, y)

        self.ax.set_title(
            f"力-位移曲线: {self.filename} - {self.current_idx}/{self.total_points} 点",
            fontsize=14,
        )
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """启动定时器并显示窗口。"""
        # 定时器以 ~30fps 刷新画面，每次推进的点数由速度滑块决定
        self.timer_interval_hz = 30
        interval_ms = int(1000 / self.timer_interval_hz)
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()

        # 默认不自动播放，让用户先查看静态图像
        self.is_playing = False
        self.btn_pause.label.set_text("播放")

        plt.show()


def main() -> None:
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("错误：当前目录没有找到 CSV 文件")
        sys.exit(1)

    # 多文件选择
    if len(csv_files) > 1:
        print(f"发现 {len(csv_files)} 个CSV文件:")
        for i, f in enumerate(csv_files):
            print(f"  [{i}] {f}")
        try:
            choice = input("请选择文件编号 (默认0): ").strip()
            idx = int(choice) if choice else 0
            if idx < 0 or idx >= len(csv_files):
                print(f"错误：编号 {idx} 超出范围，使用默认文件 0")
                idx = 0
            filepath = csv_files[idx]
        except (ValueError, KeyboardInterrupt):
            print("输入无效，使用默认文件 0")
            filepath = csv_files[0]
    else:
        filepath = csv_files[0]

    print(f"正在处理文件: {filepath}")

    try:
        time, disp, force, col_names = load_csv(filepath)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        sys.exit(1)

    if len(time) < 2:
        print("错误：数据点太少，无法计算采样率")
        sys.exit(1)

    # 估算采样率
    try:
        dt = np.median(np.diff(time))
        fs = 1.0 / dt if dt > 0 else 1.0
    except Exception as e:
        print(f"计算采样率失败: {e}")
        fs = 1.0  # 默认采样率

    if fs <= 0 or fs > 1e6:  # 合理性检查
        print(f"警告：采样率 {fs:.2f} Hz 似乎不合理，使用默认值 1.0 Hz")
        fs = 1.0

    # 应用信号处理
    try:
        disp, force = apply_signal_processing(disp, force, time, fs)
    except Exception as e:
        print(f"信号处理失败: {e}")
        print("使用原始数据继续...")
        # 使用原始数据继续

    viewer = ForceDisplacementViewer(disp, force, col_names, filename=filepath)
    viewer.run()


if __name__ == "__main__":
    main()
