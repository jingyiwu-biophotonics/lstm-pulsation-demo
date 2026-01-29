"""
Signal processing utilities: resampling, filtering, amplitude scaling, TDDR, and wavelet denoising.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
import pywt


def preprocess_signal(
    signal: np.ndarray,
    fs_signal: float,
    fs_target: float,
    amplitude_scale: float = 0.9,
    offset: float | None = None,
    plt_fig: bool = True,
):
    """
    Resample (if needed), high-pass, and scale a (1, N) signal into ~[-1,1].

    Parameters
    ----------
    signal : np.ndarray
        Input array, ideally shaped (1, N). If (N,1) or (N,), it will be flattened/reshaped.
    fs_signal : float
        Original sampling rate (Hz).
    fs_target : float
        Target sampling rate (Hz).
    amplitude_scale : float
        Target mean amplitude scaling (uses Hilbert envelope robust mean).
    offset : float or None
        DC offset to add after scaling (e.g., 0.0). If None, no offset added.
    plt_fig : bool
        If True, show diagnostic plots.

    Returns
    -------
    time_target : np.ndarray
        Time axis at target sampling rate (seconds).
    signal_target : np.ndarray
        Processed signal shaped (1, M) at fs_target.
    signal_scale : float
        Scale factor applied before offset.
    """
    s = signal
    if s.ndim > 1 and s.shape[0] != 1:
        s = s.T  # ensure (1, N)

    time_signal = np.arange(0, s.shape[1]) / fs_signal
    duration = time_signal[-1] if s.shape[1] > 1 else 0.0

    # Resample via linear interpolation if needed
    if fs_signal != fs_target:
        time_target = np.arange(0, duration, 1 / fs_target)
        interp = interp1d(time_signal, s.flatten(), kind="linear", bounds_error=False, fill_value="extrapolate")
        s_interp = interp(time_target).reshape(1, -1)
    else:
        time_target = time_signal
        s_interp = s

    # High-pass (Butterworth)
    b, a = butter(5, 0.5 / (fs_target / 2.0), btype="high")
    s_filt = filtfilt(b, a, s_interp)

    # Robust amplitude estimate via Hilbert envelope (drop 20/80 percentiles)
    env = np.abs(hilbert(s_filt))
    lo, hi = np.percentile(env, [20, 80])
    env_r = env[(env >= lo) & (env <= hi)]
    scale = np.mean(env_r) * amplitude_scale
    s_out = s_filt / (scale if scale != 0 else 1.0)
    if offset is not None:
        s_out = s_out + offset

    if plt_fig:
        plt.figure(figsize=(10, 3))
        plt.plot(time_signal, s.flatten(), label="Raw")
        plt.plot(time_target, s_interp.flatten(), label="Resampled")
        plt.plot(time_target, s_filt.flatten(), label="Filtered")
        plt.plot(time_target, s_out.flatten(), label="Scaled+Offset")
        plt.axhline(1, ls="--", c="gray")
        plt.axhline(-1, ls="--", c="gray")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("Data Preprocessing")
        plt.legend(); plt.show()

    return time_target, s_out, scale


def TDDR(signal: np.ndarray, sample_rate: float, filter_cutoff: float = 2.0):
    """
    Temporal Derivative Distribution Repair (TDDR) per Fishburn et al., NeuroImage 2019.

    Parameters
    ----------
    signal : np.ndarray
        Either (N,) or (N, C). If 2D, TDDR is applied column-wise and returns a 2D array.
    sample_rate : float
        Sampling rate (Hz).
    filter_cutoff : float
        Low-pass cutoff (Hz) for separating low/high frequency components.

    Returns
    -------
    tuple
        (signal_corrected, signal_low_corrected, signal_high, signal_mean)
        For 2D input, returns 2D arrays (per-channel).
    """
    x = np.asarray(signal)
    if x.ndim == 2:
        out = np.zeros_like(x, dtype=float)
        low_c = np.zeros_like(x, dtype=float)
        high_c = np.zeros_like(x, dtype=float)
        mean_c = np.zeros(x.shape[1], dtype=float)
        for ch in range(x.shape[1]):
            out[:, ch], low_c[:, ch], high_c[:, ch], mean_c[ch] = TDDR(x[:, ch], sample_rate, filter_cutoff)
        return out, low_c, high_c, mean_c

    x = x.astype(float)
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    m = np.mean(x)
    x = x - m

    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        low = filtfilt(fb, fa, x, padlen=0)
    else:
        low = x
    high = x - low

    # Robust derivative weighting
    deriv = np.diff(low)
    w = np.ones_like(deriv)
    mu = np.inf
    eps = np.sqrt(np.finfo(float).eps)
    tune = 4.685

    for _ in range(50):
        mu0 = mu
        mu = np.sum(w * deriv) / np.sum(w)
        dev = np.abs(deriv - mu)
        sigma = 1.4826 * np.median(dev)
        r = dev / (sigma * tune + 1e-12)
        w = ((1 - r**2) * (r < 1)) ** 2
        if abs(mu - mu0) < eps * max(abs(mu), abs(mu0)):
            break

    new_deriv = w * (deriv - mu)
    low_corr = np.cumsum(np.insert(new_deriv, 0, 0.0))
    low_corr -= np.mean(low_corr)
    corrected = low_corr + high + m
    return corrected, low_corr, high, m


def display_frequency_bands(signal: np.ndarray, sampling_frequency: float, wavelet_name: str = "db4"):
    """
    Print frequency bands per DWT level for a given signal and sampling rate.
    """
    n = len(signal)
    nyq = sampling_frequency / 2.0
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet_name).dec_len)
    print(f"Wavelet '{wavelet_name}', fs={sampling_frequency} Hz; max level={max_level}")
    for level in range(1, max_level + 1):
        f_hi = nyq / (2 ** (level - 1))
        f_lo = nyq / (2 ** level)
        print(f"Level {level}: {f_lo:.2f}–{f_hi:.2f} Hz")


def wavelet_denoise(
    signal: np.ndarray,
    sampling_frequency: float,
    selected_levels: list[int],
    wavelet_name: str = "db4",
    percentages: float | list[float] | None = None,
    reconstruct_all_bands: bool = True,
    remove_both_sides: bool = False,
) -> np.ndarray:
    """
    Denoise by zeroing large-magnitude wavelet coefficients in selected bands.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    sampling_frequency : float
        Sampling frequency in Hz (used only for reporting).
    selected_levels : list of int
        DWT levels to modify (1..max_level). Higher level = lower frequency.
    wavelet_name : str
        Wavelet family (e.g., 'db4').
    percentages : float or list[float] or None
        Percent of largest |coeff| to zero in each selected band. If float, reused.
    reconstruct_all_bands : bool
        If False, zero-out all non-selected bands when reconstructing.
    remove_both_sides : bool
        If True, also remove coefficients below a symmetric low threshold.

    Returns
    -------
    np.ndarray
        Denoised signal (same length as input).
    """
    n = len(signal)
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet_name).dec_len)
    coeffs = pywt.wavedec(signal, wavelet_name, level=max_level)

    if percentages is None:
        percentages = [5.0] * len(selected_levels)
    if isinstance(percentages, (int, float)):
        percentages = [float(percentages)] * len(selected_levels)
    if len(percentages) != len(selected_levels):
        raise ValueError("length(percentages) must match length(selected_levels)")

    nyq = sampling_frequency / 2.0
    print(f"Wavelet='{wavelet_name}', fs={sampling_frequency} Hz")
    for lvl in selected_levels:
        f_hi = nyq / (2 ** (lvl - 1))
        f_lo = nyq / (2 ** lvl)
        print(f"  Level {lvl}: {f_lo:.2f}–{f_hi:.2f} Hz")

    # coeffs ordering: [cA_L, cD_L, cD_{L-1}, ..., cD1]
    band_indices = [max_level + 1 - L for L in selected_levels][::-1]

    for pct, idx in zip(percentages, band_indices):
        band = coeffs[idx]
        thr_hi = np.percentile(np.abs(band), 100.0 - pct)
        if remove_both_sides:
            thr_lo = np.percentile(np.abs(band), pct)
            mask = (np.abs(band) > thr_hi) | (np.abs(band) < thr_lo)
            band[mask] = 0
        else:
            band[np.abs(band) > thr_hi] = 0
        coeffs[idx] = band

    if not reconstruct_all_bands:
        coeffs = [coeffs[i] if i in band_indices else np.zeros_like(coeffs[i]) for i in range(len(coeffs))]

    return pywt.waverec(coeffs, wavelet_name)[:n]
