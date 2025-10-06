"""
Windowing utilities: segment a 1×N time trace and recombine overlapped segments.
"""

from __future__ import annotations
import numpy as np
from typing import List


def segment_time_trace(time_trace: np.ndarray, window_size: int, overlap_size: int) -> List[np.ndarray]:
    """
    Segment a (1, N) time trace into overlapped windows with zero-padding on the last window if needed.

    Parameters
    ----------
    time_trace : np.ndarray
        Array shaped (1, N). If shaped (N,), reshape to (1, N) before calling.
    window_size : int
        Window length in samples.
    overlap_size : int
        Overlap between consecutive windows in samples. Must be < window_size.

    Returns
    -------
    List[np.ndarray]
        List of segments, each shaped (1, window_size).
    """
    if time_trace.ndim != 2 or time_trace.shape[0] != 1:
        raise ValueError("time_trace must be shaped (1, N).")
    n = time_trace.shape[1]
    stride = window_size - overlap_size
    if stride <= 0:
        raise ValueError("overlap_size must be < window_size.")

    segments: List[np.ndarray] = []

    if n <= window_size:
        pad_width = window_size - n
        seg = np.pad(time_trace, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
        segments.append(seg)
        return segments

    for start in range(0, n, stride):
        end = start + window_size
        seg = time_trace[:, start:end]
        if seg.shape[1] < window_size:
            pad = window_size - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)), mode="constant", constant_values=0)
        segments.append(seg)
        if end >= n:
            break

    return segments


def combine_processed_segments(
    processed_segments: list[np.ndarray],
    window_size: int,
    overlap_size: int,
    time_trace_length: int,
) -> np.ndarray:
    """
    Recombine windowed outputs (1, W) into a single (1, N) trace by averaging overlap.

    Parameters
    ----------
    processed_segments : list of np.ndarray
        List where each entry is shaped (1, window_size).
    window_size : int
        Window length in samples.
    overlap_size : int
        Overlap in samples used during segmentation.
    time_trace_length : int
        Target length (N) of the reconstructed trace (truncates tail padding).

    Returns
    -------
    np.ndarray
        Recombined trace shaped (1, N).
    """
    stride = window_size - overlap_size
    full_length = (len(processed_segments) - 1) * stride + window_size
    combined = np.zeros((1, full_length), dtype=float)
    count = np.zeros((1, full_length), dtype=float)

    for i, seg in enumerate(processed_segments):
        start = i * stride
        end = start + window_size
        combined[:, start:end] += seg
        count[:, start:end] += 1.0

    combined /= np.clip(count, 1.0, None)
    return combined[:, :time_trace_length]
