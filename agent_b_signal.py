"""
Agent B：信号处理（Series → Series）
- 低通：scipy.signal.butter + filtfilt
- 异常值：默认 IQR，可选 zscore / hampel
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

OutlierMethod = Literal["iqr", "zscore", "hampel"]


def _outliers_iqr(s: pd.Series, k: float) -> pd.Series:
    x = s.astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = x.copy()
    out[(x < lo) | (x > hi)] = np.nan
    return out.interpolate(method="linear").bfill().ffill()


def _outliers_zscore(s: pd.Series, thr: float) -> pd.Series:
    x = s.astype(float)
    mu, sigma = float(x.mean()), float(x.std(ddof=0))
    if sigma == 0 or np.isnan(sigma):
        return x
    z = np.abs((x - mu) / sigma)
    out = x.copy()
    out[z > thr] = np.nan
    return out.interpolate(method="linear").bfill().ffill()


def _outliers_hampel(s: pd.Series, window: int, n_sigma: float) -> pd.Series:
    """Hampel滤波器：使用滑动中位数和MAD检测异常值（向量化实现）"""
    x = s.astype(float).values
    n = len(x)

    # 确保窗口大小为奇数且合理
    window = max(3, window)
    if window % 2 == 0:
        window += 1  # 确保奇数窗口

    # 计算滑动中位数
    # mode='reflect' 处理边界，相当于原代码的边界处理
    median_vals = median_filter(x, size=window, mode='reflect')

    # 计算绝对偏差
    abs_dev = np.abs(x - median_vals)

    # 计算滑动MAD（中位数绝对偏差）
    # 注意：median_filter 不能直接用于 abs_dev，因为我们需要的是 abs_dev 在窗口内的中位数
    # 对于大数据集，我们可以使用卷积近似或保持循环
    # 这里使用优化后的循环，但比原代码快

    half = window // 2
    mad_vals = np.zeros_like(x)

    # 预计算一些值以加速
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # 使用预计算的绝对偏差
        mad_vals[i] = np.median(abs_dev[lo:hi])

    # 计算尺度
    scale = 1.4826 * mad_vals
    # 避免除零
    scale[scale <= 1e-12] = np.inf

    # 检测异常值
    is_outlier = np.abs(x - median_vals) > n_sigma * scale
    out = x.copy()
    out[is_outlier] = np.nan

    return pd.Series(out, index=s.index).interpolate(method="linear").bfill().ffill()


def _apply_outliers(
    s: pd.Series,
    method: OutlierMethod,
    iqr_k: float,
    zscore_thresh: float,
    hampel_window: int,
    hampel_n_sigma: float,
) -> pd.Series:
    if method == "iqr":
        return _outliers_iqr(s, iqr_k)
    if method == "zscore":
        return _outliers_zscore(s, zscore_thresh)
    return _outliers_hampel(s, hampel_window, hampel_n_sigma)


def _butter_lowpass(s: pd.Series, fs: float, cutoff_hz: float, order: int) -> pd.Series:
    nyq = fs / 2.0
    if not (0 < cutoff_hz < nyq):
        raise ValueError(f"lowpass_cutoff 须在 (0, fs/2) 内，当前 fs={fs}, cutoff={cutoff_hz}")

    # 合理限制滤波器阶数
    # 1. 不能超过用户指定的order
    # 2. 不能超过数据长度允许的阶数（需要至少 order*3 个点用于filtfilt）
    # 3. 通常不超过10阶（高阶滤波器可能不稳定）
    max_reasonable_order = 10
    max_data_order = max(1, len(s) // 3)  # 需要至少 order*3 个点

    ord_use = min(order, max_reasonable_order, max_data_order)

    # 如果阶数太高，给出警告
    if order > max_reasonable_order:
        import warnings
        warnings.warn(
            f"滤波器阶数 {order} 过高，限制为 {max_reasonable_order} 阶以获得稳定性",
            UserWarning
        )

    b, a = butter(ord_use, cutoff_hz, btype="low", fs=fs)
    x = s.astype(float).values
    # filtfilt(method='pad') 默认要求 len(x) > 3 * max(len(a), len(b))
    edge = 3 * max(len(a), len(b))
    if len(x) <= edge:
        pad = edge - len(x) + 1
        pl, pr = pad // 2, pad - pad // 2
        xp = np.pad(x, (pl, pr), mode="reflect")
        yp = filtfilt(b, a, xp)
        y = yp[pl : pl + len(x)]
    else:
        y = filtfilt(b, a, x)
    return pd.Series(y, index=s.index, name=s.name)


def process_signal(
    series: pd.Series,
    *,
    fs: float = 1.0,
    lowpass_cutoff: float | None = None,
    lowpass_order: int = 4,
    outlier_method: OutlierMethod = "iqr",
    outliers_first: bool = True,
    skip_outliers: bool = False,
    skip_lowpass: bool = False,
    iqr_k: float = 1.5,
    zscore_thresh: float = 3.0,
    hampel_window: int = 11,
    hampel_n_sigma: float = 3.0,
) -> pd.Series:
    """
    对一维序列做异常值处理与低通滤波，返回同索引的 Series。

    Parameters
    ----------
    fs : 采样率（Hz），与 lowpass_cutoff 同单位。
    lowpass_cutoff : 低通截止频率（Hz）。默认 fs/20（且小于 Nyquist）。
    outlier_method : "iqr"（默认）| "zscore" | "hampel"
    outliers_first : True 时先异常值再低通；False 时先低通再异常值。
    skip_outliers / skip_lowpass : 跳过对应步骤。
    iqr_k / zscore_thresh / hampel_window / hampel_n_sigma : 各方法参数。
    """
    s = series.copy().astype(float)
    if s.isna().any():
        s = s.interpolate(method="linear").bfill().ffill()
    if lowpass_cutoff is None:
        lowpass_cutoff = min(fs / 20.0, fs / 2.0 * 0.99)

    def step_out():
        nonlocal s
        if not skip_outliers:
            s = _apply_outliers(
                s,
                outlier_method,
                iqr_k,
                zscore_thresh,
                hampel_window,
                hampel_n_sigma,
            )

    def step_lp():
        nonlocal s
        if not skip_lowpass:
            s = _butter_lowpass(s, fs, lowpass_cutoff, lowpass_order)

    if outliers_first:
        step_out()
        step_lp()
    else:
        step_lp()
        step_out()
    return s
