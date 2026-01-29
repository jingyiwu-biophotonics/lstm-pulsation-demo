"""
Helpers for running sequence models (e.g., LSTM) over segmented inputs.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch


def process_segments_with_lstm(
    model: torch.nn.Module,
    segments: List[np.ndarray],
    device: torch.device | str = "cpu",
) -> List[np.ndarray]:
    """
    Run a detection model over (1, W) segments and collect outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with signature: y, aux = model(x) where x=[B,W,1] and y=[B,W,1].
    segments : list of np.ndarray
        List of segments shaped (1, window_size).
    device : torch.device or str
        Device to run inference on.

    Returns
    -------
    List[np.ndarray]
        Model outputs per segment, each shaped (1, window_size).
    """
    model.eval()
    device = torch.device(device)
    processed: List[np.ndarray] = []

    with torch.no_grad():
        for seg in segments:
            seg_t = torch.from_numpy(seg).float().unsqueeze(2).to(device)  # [1,W,1]
            out, _ = model(seg_t)  # [1,W,1]
            processed.append(out.squeeze(2).cpu().numpy())  # [1,W]

    return processed
