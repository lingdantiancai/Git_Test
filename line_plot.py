import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ---------- Agent A：数据与配置 ----------
@dataclass
class DataConfig:
    """可通过 load_data_config() 从 JSON 覆盖字段。"""

    data_dir: str = "."
    csv_glob: str = "*.csv"
    encoding: str | None = None  # None 表示自动探测
    sep: str | None = None  # None 表示自动探测
    # True：按列位置取前两列作 x、y（需至少 2 列，前 3 列存在时仍只用第 1、2 列）
    use_first_two_by_index: bool = True
    x_col: str = "x"
    y_col: str = "y"
    # 当 use_first_two_by_index 为 False 时，用统一列名（重命名首行）
    rename_to_xy: bool = False
    missing: str = "drop"  # drop | ffill | fill0

    @staticmethod
    def from_file(path: str | Path) -> "DataConfig":
        p = Path(path)
        if not p.is_file():
            return DataConfig()
        text = p.read_text(encoding="utf-8")
        suf = p.suffix.lower()
        if suf in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError("使用 YAML 配置请先安装 PyYAML：pip install pyyaml") from e
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        if not isinstance(raw, dict):
            return DataConfig()
        known = {f.name for f in fields(DataConfig)}
        return DataConfig(**{k: v for k, v in raw.items() if k in known})

    @staticmethod
    def discover(base: str | Path | None = None) -> "DataConfig":
        base = Path(base or Path(__file__).resolve().parent)
        for name in ("plot_data.yaml", "plot_data.yml", "plot_data.json"):
            p = base / name
            if p.is_file():
                return DataConfig.from_file(p)
        return DataConfig()


def _glob_csvs(root: str | Path, pattern: str) -> list[Path]:
    r = Path(root).resolve()
    if not r.is_dir():
        return []
    return sorted(r.glob(pattern))


def _detect_encoding(sample: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _read_sample(path: Path, max_bytes: int = 65536) -> tuple[bytes, str]:
    raw = path.read_bytes()[:max_bytes]
    enc = _detect_encoding(raw)
    return raw, enc


def _detect_sep(text_head: str) -> str:
    try:
        return csv.Sniffer().sniff(text_head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv_df(path: Path, cfg: DataConfig) -> pd.DataFrame:
    raw, enc = _read_sample(path)
    encoding = cfg.encoding or enc
    head = raw.decode(encoding, errors="replace")[:4096]
    sep = cfg.sep or _detect_sep(head)
    return pd.read_csv(path, encoding=encoding, sep=sep)


def _apply_missing(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "ffill":
        return df.ffill()
    if how == "fill0":
        return df.fillna(0)
    return df.dropna()


def prepare_xy(df: pd.DataFrame, cfg: DataConfig) -> tuple[list[float], list[float]]:
    if cfg.use_first_two_by_index:
        if df.shape[1] < 2:
            raise ValueError("CSV 至少需要 2 列（按索引取 x、y）")
        sub = df.iloc[:, :3].copy() if df.shape[1] >= 3 else df.iloc[:, :2].copy()
        work = sub.iloc[:, [0, 1]].copy()
    else:
        cols = list(df.columns)
        if cfg.rename_to_xy and len(cols) >= 2:
            work = df.iloc[:, :2].copy()
            work.columns = [cfg.x_col, cfg.y_col]
        else:
            if cfg.x_col not in df.columns or cfg.y_col not in df.columns:
                raise ValueError(f"缺少列 {cfg.x_col!r} / {cfg.y_col!r}")
            work = df[[cfg.x_col, cfg.y_col]].copy()
    work.columns = ["_x", "_y"]
    work = _apply_missing(work, cfg.missing)
    x = work["_x"].astype(float).tolist()
    y = work["_y"].astype(float).tolist()
    return x, y


def load_xy_from_config(cfg: DataConfig) -> tuple[list[float], list[float]] | None:
    paths = _glob_csvs(cfg.data_dir, cfg.csv_glob)
    if not paths:
        return None
    df = load_csv_df(paths[0], cfg)
    return prepare_xy(df, cfg)


def prepare_three_lines(df: pd.DataFrame, cfg: DataConfig) -> tuple[list[float], list[float], list[float], list[float]]:
    if cfg.use_first_two_by_index:
        n = df.shape[1]
        if n < 2:
            raise ValueError("CSV 至少需要 2 列")
        if n >= 4:
            work = df.iloc[:, :4].copy()
            work.columns = ["_x", "_y0", "_y1", "_y2"]
        elif n == 3:
            work = df.iloc[:, :3].copy()
            work.columns = ["_x", "_y0", "_y1"]
            work["_y2"] = work["_y1"]
        else:
            work = df.iloc[:, :2].copy()
            work.columns = ["_x", "_y0"]
            work["_y1"] = work["_y0"] * 0.8
            work["_y2"] = work["_y0"] * 1.2
    else:
        if cfg.x_col not in df.columns or cfg.y_col not in df.columns:
            raise ValueError(f"缺少列 {cfg.x_col!r} / {cfg.y_col!r}")
        work = df[[cfg.x_col, cfg.y_col]].copy()
        work.columns = ["_x", "_y0"]
        work["_y1"] = work["_y0"] * 0.8
        work["_y2"] = work["_y0"] * 1.2
    work = _apply_missing(work, cfg.missing)
    x = work["_x"].astype(float).tolist()
    y0 = work["_y0"].astype(float).tolist()
    y1 = work["_y1"].astype(float).tolist()
    y2 = work["_y2"].astype(float).tolist()
    return x, y0, y1, y2


def load_three_lines_from_config(cfg: DataConfig) -> tuple[list[float], list[float], list[float], list[float]] | None:
    paths = _glob_csvs(cfg.data_dir, cfg.csv_glob)
    if not paths:
        return None
    df = load_csv_df(paths[0], cfg)
    return prepare_three_lines(df, cfg)


# 配置：同目录下 plot_data.yaml / plot_data.yml / plot_data.json（任选其一）
_CFG = DataConfig.discover()
_LINES = load_three_lines_from_config(_CFG)

# 手动改 y 轴范围；任一端可为 None（该端仍由数据自动留白）
Y_MIN, Y_MAX = -0.5, 6

# Agent C：动画按块推进；每帧显示的点数步长（1 为逐点）；导出为 None 或 "xxx.gif" / "xxx.mp4"
ANIM_BLOCK = 1
ANIM_INTERVAL_MS = 80
ANIM_OUT: str | None = None

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.30)

if _LINES is not None:
    x, y0, y1, y2 = _LINES
else:
    x = [0, 1, 2, 3, 4]
    y0 = [0, 5, 1, 3, 2]
    y1 = [1, 3, 2, 4, 1]
    y2 = [2, 2, 3, 1, 4]

ys_all = [y0, y1, y2]
labels = ["曲线 1", "曲线 2", "曲线 3"]
styles = ["-o", "-s", "-^"]
line_objs = []
for lab, st in zip(labels, styles):
    (ln,) = ax.plot([], [], st, markersize=6, label=lab)
    line_objs.append(ln)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("折线图（三条线动画）")
ax.legend(loc="upper right")

flat_y = [v for row in ys_all for v in row]
y_lo, y_hi = min(flat_y), max(flat_y)
pad = (y_hi - y_lo) * 0.1 or 1.0
auto_lo, auto_hi = y_lo - pad, y_hi + pad
init_lo = Y_MIN if Y_MIN is not None else auto_lo
init_hi = Y_MAX if Y_MAX is not None else auto_hi

x_lo, x_hi = min(x), max(x)
xpad = (x_hi - x_lo) * 0.1 or 1.0
x_init_lo, x_init_hi = x_lo - xpad, x_hi + xpad

ax_y_max = fig.add_axes([0.15, 0.07, 0.7, 0.03])
ax_y_min = fig.add_axes([0.15, 0.12, 0.7, 0.03])
ax_x_min = fig.add_axes([0.15, 0.17, 0.7, 0.03])
ax_x_max = fig.add_axes([0.15, 0.22, 0.7, 0.03])

y_slider_lo = Slider(ax_y_min, "Y 下限", -10, 10, valinit=init_lo, valstep=0.1)
y_slider_hi = Slider(ax_y_max, "Y 上限", -10, 10, valinit=init_hi, valstep=0.1)
x_slider_lo = Slider(ax_x_min, "x 下限", -50, 50, valinit=x_init_lo, valstep=0.1)
x_slider_hi = Slider(ax_x_max, "x 上限", -50, 50, valinit=x_init_hi, valstep=0.1)


def on_y_slider(_):
    lo, hi = sorted((y_slider_lo.val, y_slider_hi.val))
    if hi - lo < 1e-6:
        hi = lo + 0.01
    ax.set_ylim(lo, hi)
    fig.canvas.draw_idle()


def on_x_slider(_):
    xlo, xhi = sorted((x_slider_lo.val, x_slider_hi.val))
    if xhi - xlo < 1e-6:
        xhi = xlo + 0.01
    ax.set_xlim(xlo, xhi)
    fig.canvas.draw_idle()


y_slider_lo.on_changed(on_y_slider)
y_slider_hi.on_changed(on_y_slider)
x_slider_lo.on_changed(on_x_slider)
x_slider_hi.on_changed(on_x_slider)
ax.set_ylim(init_lo, init_hi)
ax.set_xlim(x_init_lo, x_init_hi)

_n = len(x)
_n_frames = max(1, math.ceil(_n / ANIM_BLOCK)) if ANIM_BLOCK > 0 else 1


def _anim_update(frame: int):
    end = min((frame + 1) * ANIM_BLOCK, _n)
    xs = x[:end]
    for ln, yrow in zip(line_objs, ys_all):
        ln.set_data(xs, yrow[:end])
    return line_objs


_anim = FuncAnimation(
    fig,
    _anim_update,
    frames=_n_frames,
    interval=ANIM_INTERVAL_MS,
    blit=False,
    repeat=True,
)

if ANIM_OUT:
    _out = Path(ANIM_OUT)
    _fps = max(1.0, 1000.0 / ANIM_INTERVAL_MS)
    _suf = _out.suffix.lower()
    if _suf == ".gif":
        _anim.save(str(_out), writer="pillow", fps=_fps)
    elif _suf == ".mp4":
        _anim.save(str(_out), writer="ffmpeg", fps=_fps)

plt.show()
