# Force-Displacement Signal Processing & Visualization

A Python tool for processing and visualizing force-displacement test data from CSV files. It provides real-time animated plotting with a sliding window, integrated with a signal processing pipeline for outlier removal and low-pass filtering.

## Features

- **Signal Processing Pipeline** — Outlier detection (IQR / Z-score / Hampel) + Butterworth low-pass filter, configurable processing order
- **Dynamic Visualization** — Animated sliding-window plot with real-time axis scaling
- **Interactive Controls** — Speed slider (1-100 fps) and pause/resume button
- **Auto CSV Detection** — Automatically scans for CSV files in the working directory
- **Chinese Font Support** — Configured for Microsoft YaHei / SimHei out of the box

## Project Structure

```
.
├── agent_b_signal.py   # Signal processing library
├── line_plot.py        # Animated visualization with GUI controls
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV file in the project directory. Expected CSV format:

```csv
Time,Displacement,Force
(s),(mm),(N)
0.0000,0.0001,0.0011
0.0200,0.0002,0.0010
...
```

- Row 1: Column names
- Row 2: Units (skipped automatically)
- Row 3+: Numeric data

### 3. Run

```bash
python line_plot.py
```

## Signal Processing API

`agent_b_signal.py` can be used independently:

```python
import pandas as pd
from agent_b_signal import process_signal

# Create a noisy signal
s = pd.Series([...])

# Default: IQR outlier removal + Butterworth low-pass filter
processed = process_signal(s, fs=50.0)

# Z-score method, low-pass cutoff at 2 Hz
processed = process_signal(s, fs=50.0, outlier_method="zscore", lowpass_cutoff=2.0)

# Hampel filter, process low-pass first then outliers
processed = process_signal(s, fs=50.0, outlier_method="hampel", outliers_first=False)

# Skip outlier removal, only apply low-pass filter
processed = process_signal(s, fs=50.0, skip_outliers=True)
```

### `process_signal` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | `pd.Series` | — | Input signal (required) |
| `fs` | `float` | `1.0` | Sampling rate (Hz) |
| `lowpass_cutoff` | `float` | `fs/20` | Low-pass cutoff frequency (Hz) |
| `lowpass_order` | `int` | `4` | Butterworth filter order |
| `outlier_method` | `"iqr"` \| `"zscore"` \| `"hampel"` | `"iqr"` | Outlier detection method |
| `outliers_first` | `bool` | `True` | Process outliers before low-pass if True |
| `skip_outliers` | `bool` | `False` | Skip outlier removal |
| `skip_lowpass` | `bool` | `False` | Skip low-pass filtering |
| `iqr_k` | `float` | `1.5` | IQR multiplier (for `method="iqr"`) |
| `zscore_thresh` | `float` | `3.0` | Z-score threshold (for `method="zscore"`) |
| `hampel_window` | `int` | `11` | Window size (for `method="hampel"`) |
| `hampel_n_sigma` | `float` | `3.0` | MAD sigma threshold (for `method="hampel"`) |

## Dependencies

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)

## License

This project is for educational and research purposes.
