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
    x = s.astype(float).values
    n = len(x)
    half = max(1, window // 2)
    out = x.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            continue
        if np.abs(x[i] - med) > n_sigma * scale:
            out[i] = np.nan
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
    ord_use = min(order, max(1, len(s) // 4))
    b, a = butter(ord_use, cutoff_hz, btype="low", fs=fs)
    x = s.astype(float).values
    # filtfilt(method='pad') 默认要求 len(x) > 3 * max(len(a), len(b))
    edge = 3 * max(len(a), len(b))
    if len(x) <= edge:
        pad = edge - len(x) + 1
        pl, pr = pad // 2, pad - pad // 2
        xp = np.pad(x, (pl, pr), mode="edge")
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
