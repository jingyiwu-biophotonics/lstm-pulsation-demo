"""
Pulse detection and analysis: peaks/valleys, segmentation, and statistics.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple


def find_peaks_and_valleys(
    processed_signal: np.ndarray,
    prominence: float | None = None,
    height: float | None = None,
    distance: int | None = None,
):
    """
    Find peaks and valleys (local minima) in a 1D processed signal.

    Returns
    -------
    peaks : np.ndarray
        Indices of peaks.
    valleys : np.ndarray
        Indices of valleys (via peak-finding on the inverted signal).
    """
    peaks, _ = find_peaks(processed_signal, prominence=prominence, height=height, distance=distance)
    valleys, _ = find_peaks(-processed_signal, prominence=prominence, height=height, distance=distance)
    return peaks, valleys


def segment_pulses(signal: np.ndarray, valleys: np.ndarray) -> List[np.ndarray]:
    """
    Segment signal between consecutive valleys (excludes the end index).
    """
    segs: List[np.ndarray] = []
    for i in range(len(valleys) - 1):
        start, end = valleys[i], valleys[i + 1]
        segs.append(signal[start:end])
    return segs


def calculate_weighted_mean_std(segmented_pulses: List[np.ndarray], weights: np.ndarray | None = None):
    """
    Weighted mean and std over variable-length pulses (NaN-padded alignment).

    Returns
    -------
    mean_pulse : np.ndarray
    std_pulse : np.ndarray
    """
    max_len = max(len(p) for p in segmented_pulses)
    padded = np.array([np.pad(p, (0, max_len - len(p)), constant_values=np.nan) for p in segmented_pulses])

    if weights is None:
        weights = np.ones(len(segmented_pulses), dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    mean_pulse = np.nansum(padded * w[:, None], axis=0)
    var_pulse = np.nansum(w[:, None] * (padded - mean_pulse) ** 2, axis=0)
    std_pulse = np.sqrt(var_pulse)
    return mean_pulse, std_pulse


def z_score_filter(segmented_pulses: List[np.ndarray], z_threshold: float = 2.0):
    """
    Filter pulses using a global z-score vs. the mean template (NaN-aware).

    Returns
    -------
    filtered_pulses : list[np.ndarray]
        Pulses that pass the z-score threshold (returned padded to max length).
    weights : list[float]
        1 for included pulses, 0 for excluded.
    """
    max_len = max(len(p) for p in segmented_pulses)
    padded = np.array([np.pad(p, (0, max_len - len(p)), constant_values=np.nan) for p in segmented_pulses])

    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    std[std == 0] = np.nan

    filtered, weights = [], []
    for p in padded:
        z = (p - mean) / std
        if np.nanmax(np.abs(z)) <= z_threshold:
            filtered.append(p)
            weights.append(1.0)
        else:
            weights.append(0.0)

    return filtered, weights

def segmentation_and_processing(time_trace, valleys, fs_target, z_threshold=2):
    """
    Segments pulses , filters outliers,
    computes filtered mean/std

    Returns
    -------
    results : dict
        {
            "x_pulse": ...,
            "raw_pulse" ...,
            "filtered_pulse" ...,
            "filtered_mean_pulse": ...,
            "filtered_std_pulse"
        }
    """

    raw_pulse     = segment_pulses(np.asarray(time_trace).squeeze(), valleys) # Segment pulses
    filt_pulse, _     = z_score_filter(raw_pulse,     z_threshold=z_threshold) # Z-score filtering (remove outlier beats)
    fmean_pulse,     fstd_pulse     = calculate_weighted_mean_std(filt_pulse) # Pulse mean/std
    x_pulse     = np.arange(len(fmean_pulse))     / float(fs_target) # Time axis for pulses (seconds)

    # Collect results
    results = {
        "x_pulse": x_pulse,
        "raw_pulse": raw_pulse,
        "filtered_pulse": filt_pulse,
        "filtered_mean_pulse": fmean_pulse,
        "filtered_std_pulse": fstd_pulse,
    }
    return results

def interp_to_common_axis(t1, y1, t2, y2, time_range, fs):
    """
    This function interpolates both singal/pulse onto a common uniform time axis over a given time window at
    sampling frequency fs.

    Parameters
    ----------
    t1, y1 : array-like
        Time axis and magnitudes for trace 1.
    t2, y2 : array-like
        Time axis and magnitudes for trace 2.
    time_range : tuple (t_start, t_end)
        Window [t_start, t_end] (in seconds) within which to compute correlation.
    fs : float
        Sampling frequency (Hz) for the common axis used in interpolation.

    Returns
    -------
    t_common : np.ndarray
        Common time axis (seconds) used for correlation.
    y1_common : np.ndarray
        Interpolated values of trace 1 on t_common.
    y2_common : np.ndarray
        Interpolated values of trace 2 on t_common.
    """
    # ---- sanitize inputs ----
    t1 = np.asarray(t1, float).ravel()
    y1 = np.asarray(y1, float).ravel()
    t2 = np.asarray(t2, float).ravel()
    y2 = np.asarray(y2, float).ravel()

    if t1.size < 2 or t2.size < 2 or fs is None or fs <= 0:
        return np.nan, np.array([]), np.array([]), np.array([])

    # Ensure ascending time (if not already)
    if not np.all(np.diff(t1) > 0):
        p = np.argsort(t1)
        t1, y1 = t1[p], y1[p]
    if not np.all(np.diff(t2) > 0):
        p = np.argsort(t2)
        t2, y2 = t2[p], y2[p]

    # ---- restrict to requested window ----
    t0_req, t1_req = float(time_range[0]), float(time_range[1])
    if not (t1_req > t0_req):
        return np.nan, np.array([]), np.array([]), np.array([])

    m1 = (t1 >= t0_req) & (t1 <= t1_req)
    m2 = (t2 >= t0_req) & (t2 <= t1_req)
    if not np.any(m1) or not np.any(m2):
        return np.nan, np.array([]), np.array([]), np.array([])

    t1_w, y1_w = t1[m1], y1[m1]
    t2_w, y2_w = t2[m2], y2[m2]

    # ---- true overlap inside the window (prevents extrapolation) ----
    t_lo = max(t1_w[0], t2_w[0], t0_req)
    t_hi = min(t1_w[-1], t2_w[-1], t1_req)
    if not (t_hi > t_lo):
        return np.nan, np.array([]), np.array([]), np.array([])

    # ---- build common uniform time axis ----
    dt = 1.0 / float(fs)
    # include endpoint if it lands within half a step
    n_steps = int(np.floor((t_hi - t_lo) / dt))
    t_common = t_lo + np.arange(n_steps + 1) * dt
    if t_common.size < 2:
        return np.nan, np.array([]), np.array([]), np.array([])

    # ---- interpolate both onto common axis (no extrapolation due to overlap choice) ----
    y1_common = np.interp(t_common, t1_w, y1_w)
    y2_common = np.interp(t_common, t2_w, y2_w)

    return t_common, y1_common, y2_common

def pearson_r(a, b):
    """Compute Pearson correlation coefficient between two arrays."""
    return np.corrcoef(a, b)[0, 1]


def extract_ibi_hr(signal, fs, height=None, distance=None, prominence=None):
    """
    Extract Inter-Beat Intervals (IBI) and Heart Rate (HR) from a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (clean/processed).
    fs : float
        Sampling frequency in Hz.
    height, distance, prominence : optional
        Parameters for peak detection.

    Returns
    -------
    peak_times : np.ndarray
        Times of detected peaks.
    peak_heights : np.ndarray
        Heights of detected peaks.
    ibi : np.ndarray
        Inter-beat intervals in seconds.
    hr : np.ndarray
        Instantaneous heart rate in Hz.
    """
    time = np.arange(len(signal.squeeze())) / fs
    peaks_indices, _ = find_peaks(signal, prominence=prominence, height=height, distance=distance)

    peak_times = time[peaks_indices]
    peak_heights = signal[peaks_indices]

    ibi = np.diff(peak_times)
    hr = 1 / ibi

    return peak_times, peak_heights, ibi, hr