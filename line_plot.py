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
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from agent_b_signal import process_signal

# ---------- 常量 ----------
WINDOW_SIZE = 20
DEFAULT_SPEED = 30  # 帧/秒


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
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.disp = disp
        self.force = force
        self.col_names = col_names
        self.filename = filename
        self.window_size = window_size
        self.total_points = len(disp)
        self.is_playing = True

        # 固定坐标轴范围（基于完整数据）
        self.x_min, self.x_max = disp.min(), disp.max()
        self.y_min, self.y_max = force.min(), force.max()

        self._build_ui()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)

        (self.line,) = self.ax.plot([], [], "b-", linewidth=1.5, marker="o", markersize=3)

        self.ax.set_xlabel(f"{self.col_names[1]} (位移)", fontsize=13)
        self.ax.set_ylabel(f"{self.col_names[2]} (力)", fontsize=13)
        self.ax.set_title(
            f"力-位移曲线: {self.filename} (窗口大小: {self.window_size})", fontsize=14
        )
        self.ax.grid(True, alpha=0.3)

        # 固定坐标轴范围
        pad_x = (self.x_max - self.x_min) * 0.05 if self.x_max > self.x_min else 0.5
        pad_y = (self.y_max - self.y_min) * 0.05 if self.y_max > self.y_min else 0.5
        self.ax.set_xlim(self.x_min - pad_x, self.x_max + pad_x)
        self.ax.set_ylim(self.y_min - pad_y, self.y_max + pad_y)

        # 速度滑块
        ax_speed = self.fig.add_axes([0.15, 0.10, 0.5, 0.03])
        self.slider_speed = Slider(
            ax_speed, "速度 (帧/秒)", 1, 100, valinit=DEFAULT_SPEED, valstep=1
        )

        # 播放/暂停按钮
        ax_pause = self.fig.add_axes([0.70, 0.09, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, "暂停")
        self.btn_pause.on_clicked(self._on_pause_clicked)

    # ---- 回调 ----

    def _on_pause_clicked(self, _event) -> None:
        self.is_playing = not self.is_playing
        self.btn_pause.label.set_text("继续" if not self.is_playing else "暂停")
        self.fig.canvas.draw_idle()

    def _get_interval(self) -> float:
        return 1000.0 / self.slider_speed.val

    # ---- 动画 ----

    def _update(self, frame: int) -> tuple:
        if not self.is_playing:
            return (self.line,)

        end_idx = frame + 1
        start_idx = max(0, end_idx - self.window_size)

        x = self.disp[start_idx:end_idx]
        y = self.force[start_idx:end_idx]

        self.line.set_data(x, y)

        self.ax.set_title(
            f"力-位移曲线: {self.filename} - 已加载: {end_idx}/{self.total_points} 点",
            fontsize=14,
        )
        return (self.line,)

    def run(self) -> None:
        """启动动画并显示窗口。"""
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            frames=self.total_points,
            interval=self._get_interval(),
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        self.slider_speed.on_changed(
            lambda _val: setattr(self.ani.event_source, "interval", self._get_interval())
        )
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
