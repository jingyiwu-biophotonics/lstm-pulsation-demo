"""
Visualization utilities (Plotly and Matplotlib) for segments, comparisons, spectrograms, and SQI.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_segments(time_trace: np.ndarray, segments: list[np.ndarray], window_size: int, overlap_size: int, title: str, sampling_rate: float = 50.0):
    """
    Plot original (1, N) time trace and each overlapped segment on separate subplot rows (Plotly).
    """
    n = time_trace.shape[1]
    stride = window_size - overlap_size
    num_segments = len(segments)

    fig = make_subplots(
        rows=num_segments + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Original Time Trace ({n} samples)"]
        + [f"Segment {i+1} ({segments[i].shape[1]} samples)" for i in range(num_segments)],
    )

    t_full = np.arange(n) / sampling_rate
    fig.add_trace(go.Scatter(x=t_full, y=time_trace[0], mode="lines", name="Original", line=dict(color="orange")), row=1, col=1)

    for i, seg in enumerate(segments):
        start = i * stride
        t_seg = np.arange(start, start + window_size) / sampling_rate
        fig.add_trace(go.Scatter(x=t_seg, y=seg[0], mode="lines", name=f"Segment {i+1}", line=dict(color="orange")), row=i + 2, col=1)

    fig.update_layout(height=300 * (num_segments + 1), width=1000, title_text=title, showlegend=False)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    fig.show()


def plot_segments_comparison(
    time_trace: np.ndarray,
    processed_time_trace: np.ndarray,
    original_segments: list[np.ndarray],
    processed_segments: list[np.ndarray],
    window_size: int,
    overlap_size: int,
    title: str,
    plot_segments: bool = True,
    sampling_rate: float = 50.0,
):
    """
    Compare full original vs processed trace, and optionally each segment (Plotly).
    """
    n = time_trace.shape[1]
    stride = window_size - overlap_size
    num_segments = len(original_segments)
    rows = num_segments + 1 if plot_segments else 1

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=["Full Time Trace"] + [f"Segment {i+1}" for i in range(num_segments)])

    t_full = np.arange(n) / sampling_rate
    t_proc = np.arange(processed_time_trace.shape[1]) / sampling_rate
    fig.add_trace(go.Scatter(x=t_full, y=time_trace[0], mode="lines", name="Original", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_proc, y=processed_time_trace[0], mode="lines", name="Processed", line=dict(color="blue", width=0.75)), row=1, col=1)

    if plot_segments:
        for i, (orig_seg, proc_seg) in enumerate(zip(original_segments, processed_segments)):
            start = i * stride
            t_seg = np.arange(start, start + window_size) / sampling_rate
            fig.add_trace(go.Scatter(x=t_seg, y=orig_seg[0], mode="lines", name=f"Original {i+1}", line=dict(color="orange")), row=i + 2, col=1)
            fig.add_trace(go.Scatter(x=t_seg, y=proc_seg[0], mode="lines", name=f"Processed {i+1}", line=dict(color="blue", width=0.75)), row=i + 2, col=1)

    fig.update_layout(height=400 * (num_segments + 1 if plot_segments else 1), width=1000, title_text=title, showlegend=False)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    fig.show()


def plot_spectrogram(signal: np.ndarray, fs: float, nperseg_ratio: float = 0.075, noverlap_ratio: float = 0.9, title_text: str = "Spectrogram"):
    """
    Matplotlib spectrogram from a 1D signal.
    """
    sig = signal.flatten()
    nperseg = max(8, int(len(sig) * nperseg_ratio))
    noverlap = int(nperseg * noverlap_ratio)

    f, t, Sxx = spectrogram(sig, fs, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(5, 2))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    plt.colorbar(label="Power (dB)")
    plt.ylim([0, 5])
    plt.title(title_text)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def calculate_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    SNR in dB, using clean as reference.
    """
    sp = np.mean(clean_signal**2)
    npow = np.mean((noisy_signal - clean_signal) ** 2)
    return 10.0 * np.log10(sp / (npow + 1e-12))


def plot_sqi_results_snr(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    window_size: int,
    step_size: int,
    threshold: float | None = None,
    fs: float = 50.0,
):
    """
    Plot noisy vs processed signal and a moving-window SNR curve with low-SNR shading (Plotly).
    """
    snr_vals = []
    for i in range(0, len(noisy_signal) - window_size + 1, step_size):
        snr_vals.append(calculate_snr(clean_signal[i : i + window_size], noisy_signal[i : i + window_size]))
    snr_vals = np.asarray(snr_vals)
    t = np.arange(len(noisy_signal)) / fs
    t_snr = np.arange(0, len(noisy_signal) - window_size + 1, step_size) / fs
    thr = np.mean(snr_vals) * 0.75 if threshold is None else threshold

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.5, 0.25], subplot_titles=["Signals", "SNR vs Time"]
    )
    fig.add_trace(go.Scatter(x=t, y=noisy_signal, mode="lines", name="Original", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=clean_signal, mode="lines", name="Processed", line=dict(color="blue")), row=1, col=1)

    for idx, val in enumerate(snr_vals):
        if val < thr:
            start = t_snr[idx]
            end = start + window_size / fs
            fig.add_vrect(x0=start, x1=end, fillcolor="rgba(255,0,0,0.1)", line_width=0)

    t_snr_center = t_snr + window_size / (2 * fs)
    fig.add_trace(go.Scatter(x=t_snr_center, y=snr_vals, mode="lines", name="SNR", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_snr_center, y=[thr] * len(snr_vals), mode="lines", name="Threshold", line=dict(color="black", dash="dash")), row=2, col=1)

    fig.update_layout(height=600, width=1200, title_text="Signal Quality Index (SNR)", showlegend=True)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="SNR (dB)", row=2, col=1)
    fig.show()


def plot_signals_with_boundaries(
    noisy_signal: np.ndarray, processed_signal: np.ndarray, fs: float, valleys: np.ndarray, title: str = ""
):
    """
    Plot noisy/processed signals with vertical lines at valley indices (Plotly).
    """
    t = np.arange(len(noisy_signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=noisy_signal, mode="lines", name="Noisy", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=t, y=processed_signal, mode="lines", name="Clean", line=dict(color="blue")))

    y0, y1 = float(np.min(noisy_signal)), float(np.max(noisy_signal))
    for v in valleys:
        fig.add_trace(go.Scatter(x=[t[v], t[v]], y=[y0, y1], mode="lines", line=dict(color="grey", width=0.75), showlegend=False, hoverinfo="skip"))

    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude", showlegend=True)
    fig.show()


def plot_all_segmented_pulses(segmented_pulses: list[np.ndarray], title: str = "Segmented Pulses"):
    """
    Overlay all segmented pulses for quick quality inspection (Plotly).
    """
    fig = go.Figure()
    for i, p in enumerate(segmented_pulses):
        fig.add_trace(go.Scatter(y=p, mode="lines", name=f"Pulse {i+1}"))
    fig.update_layout(title=title, width=600, height=400, xaxis_title="Time (samples)", yaxis_title="Amplitude", showlegend=True)
    fig.show()

def extract_low_snr_regions(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    window_size: int,
    step_size: int,
    threshold: float = None,
    fs: float = 50.0,
    sort_region_by_snr: bool = True
) -> list[tuple[float, float]]:
    """
    Extract low SNR regions from a noisy signal based on a moving-window SNR analysis.
    The function merges overlapping or contiguous low SNR regions and optionally sorts 
    them by their minimum SNR values.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Noisy input signal (1D array).
    clean_signal : np.ndarray
        Reference clean signal (1D array).
    window_size : int
        Window size for SNR calculation (in samples).
    step_size : int
        Step size for moving window (in samples).
    threshold : float, optional
        SNR threshold. If None, defaults to 0.75 * mean(SNR values).
    fs : float, optional
        Sampling frequency in Hz (default = 50).
    sort_region_by_snr : bool, optional
        If True, regions are sorted from lowest to highest SNR. 
        If False, regions are returned in temporal order.

    Returns
    -------
    list of tuple
        List of low-SNR regions as (start_time, end_time).
    """
    # Compute SNR in moving windows
    snr_values = []
    for i in range(0, len(noisy_signal) - window_size + 1, step_size):
        snr = calculate_snr(clean_signal[i:i+window_size], noisy_signal[i:i+window_size])
        snr_values.append(snr)
    snr_values = np.array(snr_values)

    # Time axis for the start of each window
    snr_time_axis = np.arange(0, len(noisy_signal) - window_size + 1, step_size) / fs

    # Use default threshold if none is provided
    if threshold is None:
        threshold = np.mean(snr_values) * 0.75

    # Identify low SNR indices
    low_snr_indices = np.where(snr_values < threshold)[0]

    # Merge overlapping or contiguous low SNR regions
    merged_regions = []
    if len(low_snr_indices) > 0:
        start_idx = low_snr_indices[0]
        end_idx = low_snr_indices[0]

        for i in range(1, len(low_snr_indices)):
            if low_snr_indices[i] == end_idx + 1:
                end_idx = low_snr_indices[i]
            else:
                start_time = snr_time_axis[start_idx]
                end_time = snr_time_axis[end_idx] + window_size / fs
                min_snr = np.min(snr_values[start_idx:end_idx + 1])
                merged_regions.append((start_time, end_time, min_snr))
                start_idx = low_snr_indices[i]
                end_idx = low_snr_indices[i]

        # Add final region
        start_time = snr_time_axis[start_idx]
        end_time = snr_time_axis[end_idx] + window_size / fs
        min_snr = np.min(snr_values[start_idx:end_idx + 1])
        merged_regions.append((start_time, end_time, min_snr))

    # Return sorted or unsorted regions
    if not sort_region_by_snr:
        return [(start, end) for start, end, _ in merged_regions]
    else:
        merged_regions.sort(key=lambda x: x[2])  # sort by minimum SNR
        return [(start, end) for start, end, _ in merged_regions]

def plot_signal_segments(t_signal, y_signal_1, y_signal_2, segment_length=180, n=1, xlim=None):
    """
    Plots n random segments of a signal or a user-specified xlim.

    Parameters:
        t_signal (array): Time array.
        y_signal_1 (array): First signal array (e.g., smoothed HR NIRS).
        y_signal_2 (array): Second signal array (e.g., smoothed HR ECG).
        segment_length (int): Length of the segment to plot (in time units).
        n (int): Number of random segments to plot.
        xlim (tuple): Optional, set custom xlim (start, end). If provided, plots only that segment.
    """

    # Ensure t_signal and y_signals are numpy arrays
    t_signal = np.array(t_signal)
    y_signal_1 = np.array(y_signal_1)
    y_signal_2 = np.array(y_signal_2)

    # If xlim is provided, plot that specific segment
    if xlim:
        start_idx = np.searchsorted(t_signal, xlim[0])
        end_idx = np.searchsorted(t_signal, xlim[1])

        # Plot the selected segment
        plt.figure(figsize=(6, 2.5), dpi=150)
        plt.plot(t_signal[start_idx:end_idx], y_signal_1[start_idx:end_idx]*60, linewidth=1.5, color='orange', label='NIRS (LSTM)')
        plt.plot(t_signal[start_idx:end_idx], y_signal_2[start_idx:end_idx]*60, linewidth=0.5, color='blue', label='ECG (ground truth)')
        plt.xlim(xlim)
        plt.ylim([60, 100])
        plt.xlabel('Time (seconds)')
        plt.ylabel('HR (bpm)')
        plt.title('Zoomed Segment with xlim')
        plt.legend()
        plt.show()
        return

    # Randomly select `n` segments
    max_start = t_signal[-1] - segment_length
    for _ in range(n):
        # Randomly pick the start time for the segment
        start_time = np.random.uniform(t_signal[0], max_start)
        end_time = start_time + segment_length

        # Get the corresponding indices
        start_idx = np.searchsorted(t_signal, start_time)
        end_idx = np.searchsorted(t_signal, end_time)

        # Plot the selected segment
        plt.figure(figsize=(6, 2.5), dpi=150)
        plt.plot(t_signal[start_idx:end_idx], y_signal_1[start_idx:end_idx]*60, linewidth=1.5, color='orange', label='NIRS (LSTM)')
        plt.plot(t_signal[start_idx:end_idx], y_signal_2[start_idx:end_idx]*60, linewidth=0.5, color='blue', label='ECG (ground truth)')
        plt.xlim([start_time, end_time])
        plt.ylim([60, 100])
        plt.xlabel('Time (seconds)')
        plt.ylabel('HR (bpm)')
        plt.title(f'Segment from {start_time:.2f} to {end_time:.2f} seconds')
        plt.legend()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

# Example usage:
# plot_signal_segments(t_nirs, hr_nirs_smoothed, hr_ecg_smoothed, segment_length=180, n=3, xlim=None)
# or
# plot_signal_segments(t_nirs, hr_nirs_smoothed, hr_ecg_smoothed, segment_length=180, n=1, xlim=[3000, 3180])