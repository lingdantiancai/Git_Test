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

# ---------- 常量 ----------
DEFAULT_SPEED = 30  # 点/秒


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
    disp_processed = process_signal(disp_series, fs=fs, lowpass_cutoff=fs / 20).values
    force_processed = process_signal(force_series, fs=fs, lowpass_cutoff=fs / 20).values
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

        # 每帧推进的步数 = 速度(点/秒) / 定时器频率(帧/秒)
        steps = max(1, round(self.speed / self.timer_interval_hz))
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

        # 默认开始播放
        self.is_playing = True
        self.btn_pause.label.set_text("暂停")

        plt.show()


def main() -> None:
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("错误：当前目录没有找到 CSV 文件")
        sys.exit(1)

    filepath = csv_files[0]
    time, disp, force, col_names = load_csv(filepath)

    # 估算采样率
    dt = np.median(np.diff(time))
    fs = 1.0 / dt if dt > 0 else 1.0

    # 应用信号处理
    disp, force = apply_signal_processing(disp, force, time, fs)

    viewer = ForceDisplacementViewer(disp, force, col_names, filename=filepath)
    viewer.run()


if __name__ == "__main__":
    main()
