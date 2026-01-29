"""
LSTM Pulsation Tracing - Streamlit Web Application

A web-based demo for cleaning noisy optical signals (NIRS, PPG, DCS)
to trace cardiac pulsations using a trained LSTM model.

Author: Jingyi Wu (jingyiwu@andrew.cmu.edu)
"""

import os
import io
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat

# Import utility functions
from utils import (
    LSTMDetectionModel,
    preprocess_signal,
    find_peaks_and_valleys,
    segment_pulses,
    calculate_weighted_mean_std,
    z_score_filter,
    segment_time_trace,
    combine_processed_segments,
    process_segments_with_lstm,
)

# --- Constants ---
MODEL_PATH = "models/lstm_full_dataset.pt"
EXAMPLE_DATA_DIR = "data"
FS_TARGET = 50.0  # Target sampling frequency (Hz) - matches training data
WINDOW_SIZE = 3000  # Window size for segmentation (1 min at 50 Hz)
DEFAULT_OVERLAP_SIZE = 2000  # Default overlap between windows

# Color scheme
COLOR_NOISY = "#ffaa1c"  # Orange/yellow for noisy signal
COLOR_CLEAN = "#405FC1"  # Blue for cleaned signal
COLOR_SNR = "#9E3AB7"    # Purple for SNR
COLOR_RAW = "#2ca02c"    # Green for raw input signal


@st.cache_resource
def load_model():
    """Load the trained LSTM model (cached)."""
    device = torch.device("cpu")
    model = LSTMDetectionModel(n_hid=256, n_layers=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def load_mat_file(file_content, filename):
    """Load signal data from a .mat file."""
    try:
        data = loadmat(io.BytesIO(file_content))
        signal = None

        for key in ['signal', 'Signal', 'data', 'Data', 'y', 'x']:
            if key in data:
                signal = np.array(data[key]).flatten()
                break

        if signal is None:
            for key in data.keys():
                if not key.startswith('__'):
                    val = np.array(data[key])
                    if val.ndim >= 1 and val.size > 100:
                        signal = val.flatten()
                        break

        return signal
    except Exception as e:
        st.error(f"Error loading .mat file: {e}")
        return None


def load_csv_file(file_content):
    """Load signal data from a .csv file."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        if 'signal' in df.columns:
            signal = df['signal'].values
        elif 'Signal' in df.columns:
            signal = df['Signal'].values
        else:
            signal = df.iloc[:, 0].values
        return signal.astype(float)
    except Exception as e:
        st.error(f"Error loading .csv file: {e}")
        return None


def load_txt_file(file_content):
    """Load signal data from a .txt file."""
    try:
        signal = np.loadtxt(io.BytesIO(file_content))
        return signal.flatten()
    except Exception as e:
        st.error(f"Error loading .txt file: {e}")
        return None


def convert_to_dod(signal):
    """Convert raw intensity to delta optical density (dOD)."""
    return -np.log(signal / np.nanmean(signal))


def calculate_snr_windowed(noisy_signal, clean_signal, window_size=50, step_size=25):
    """Calculate windowed SNR values."""
    snr_values = []
    for i in range(0, len(noisy_signal) - window_size + 1, step_size):
        clean_window = clean_signal[i:i + window_size]
        noisy_window = noisy_signal[i:i + window_size]

        signal_power = np.mean(clean_window ** 2)
        noise_power = np.mean((noisy_window - clean_window) ** 2)
        snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
        snr_values.append(snr)

    return np.array(snr_values)


def process_signal(model, device, raw_signal, fs_data, amplitude_scale=0.95, offset=0.1,
                   overlap_size=DEFAULT_OVERLAP_SIZE, convert_dod=True):
    """
    Full processing pipeline: preprocess -> LSTM inference -> pulse detection.
    """
    # Ensure signal is in correct shape (1, N)
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(1, -1)
    elif raw_signal.shape[0] != 1:
        raw_signal = raw_signal.T

    # Store raw signal for visualization (before any preprocessing)
    raw_signal_original = raw_signal.copy()

    # Optionally convert to dOD
    if convert_dod:
        time_trace_raw = convert_to_dod(raw_signal)
    else:
        time_trace_raw = raw_signal.copy()

    # Preprocess: resample to 50 Hz, high-pass filter, normalize
    t_target, time_trace, signal_scale = preprocess_signal(
        time_trace_raw, fs_data, FS_TARGET,
        amplitude_scale=amplitude_scale, offset=offset, plt_fig=False
    )

    # Create time axis for raw signal
    t_raw = np.arange(raw_signal_original.shape[1]) / fs_data

    # Segment the trace into overlapping windows
    segments = segment_time_trace(time_trace, WINDOW_SIZE, overlap_size)

    # Run LSTM inference on each segment
    processed_segments = process_segments_with_lstm(model, segments, device)

    # Combine processed segments
    processed_time_trace = combine_processed_segments(
        processed_segments, WINDOW_SIZE, overlap_size, time_trace.shape[1]
    )

    # Find pulse valleys on processed trace
    _, valleys = find_peaks_and_valleys(
        processed_time_trace.squeeze(), prominence=0.25, distance=FS_TARGET / 2
    )

    # Segment pulses from both noisy and processed traces
    segmented_pulses_noisy = segment_pulses(time_trace.squeeze(), valleys)
    segmented_pulses_processed = segment_pulses(processed_time_trace.squeeze(), valleys)

    # Z-score filtering to remove outliers
    filtered_pulses_noisy, _ = z_score_filter(segmented_pulses_noisy, z_threshold=2)
    filtered_pulses_processed, _ = z_score_filter(segmented_pulses_processed, z_threshold=2)

    # Calculate mean and std pulses
    if len(filtered_pulses_noisy) > 0:
        mean_pulse_noisy, std_pulse_noisy = calculate_weighted_mean_std(filtered_pulses_noisy)
        mean_pulse_processed, std_pulse_processed = calculate_weighted_mean_std(filtered_pulses_processed)
    else:
        mean_pulse_noisy, std_pulse_noisy = calculate_weighted_mean_std(segmented_pulses_noisy)
        mean_pulse_processed, std_pulse_processed = calculate_weighted_mean_std(segmented_pulses_processed)

    # Calculate metrics
    x_pulse = np.arange(len(mean_pulse_noisy)) / FS_TARGET

    pulse_raw = np.asarray(mean_pulse_noisy).ravel()
    pulse_lstm = np.asarray(mean_pulse_processed).ravel()

    # Pearson correlation
    r_pearson = np.corrcoef(pulse_raw, pulse_lstm)[0, 1]

    # Time-to-peak difference
    ttp_raw = x_pulse[np.argmax(pulse_raw)]
    ttp_lstm = x_pulse[np.argmax(pulse_lstm)]
    delta_ttp = abs(ttp_lstm - ttp_raw)

    # Calculate windowed SNR
    snr_values = calculate_snr_windowed(
        time_trace.squeeze(), processed_time_trace.squeeze(),
        window_size=50, step_size=25
    )

    return {
        't': t_target,
        't_raw': t_raw,
        'raw_input': raw_signal_original.squeeze(),
        'noisy': time_trace.squeeze(),
        'processed': processed_time_trace.squeeze(),
        'valleys': valleys,
        'x_pulse': x_pulse,
        'mean_pulse_noisy': mean_pulse_noisy,
        'std_pulse_noisy': std_pulse_noisy,
        'mean_pulse_processed': mean_pulse_processed,
        'std_pulse_processed': std_pulse_processed,
        'ttp_raw': ttp_raw,
        'ttp_lstm': ttp_lstm,
        'r_pearson': r_pearson,
        'delta_ttp': delta_ttp,
        'snr_values': snr_values,
        'fs': FS_TARGET,
        'fs_original': fs_data,
    }


def create_signal_plot(results):
    """Create the main signal comparison plot with raw input, processed signals, and SNR panel."""
    t = results['t']
    t_raw = results['t_raw']
    raw_input = results['raw_input']
    noisy = results['noisy']
    processed = results['processed']
    snr_values = results['snr_values']

    # Calculate SNR time axis
    snr_step = 25
    snr_window = 50
    t_snr = np.arange(0, len(noisy) - snr_window + 1, snr_step) / results['fs']
    t_snr_center = t_snr + snr_window / (2 * results['fs'])

    # Create subplot figure with 3 rows: raw input, signal comparison, SNR
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.25, 0.5, 0.25],
        subplot_titles=["Raw Input Signal", "Signal Comparison (after preprocessing)", "Signal-to-Noise Ratio (SNR)"]
    )

    # Row 1: Raw input signal
    fig.add_trace(
        go.Scatter(
            x=t_raw, y=raw_input,
            mode='lines',
            name='Raw Input',
            line=dict(color=COLOR_RAW, width=1),
            showlegend=True,
        ),
        row=1, col=1
    )

    # Row 2: Noisy (preprocessed) signal
    fig.add_trace(
        go.Scatter(
            x=t, y=noisy,
            mode='lines',
            name='Original (Noisy)',
            line=dict(color=COLOR_NOISY, width=1),
        ),
        row=2, col=1
    )

    # Row 2: Processed signal
    fig.add_trace(
        go.Scatter(
            x=t, y=processed,
            mode='lines',
            name='LSTM Cleaned',
            line=dict(color=COLOR_CLEAN, width=1.2),
        ),
        row=2, col=1
    )

    # Row 3: SNR curve
    snr_plot = snr_values[:len(t_snr_center)]
    if len(snr_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=t_snr_center, y=snr_plot,
                mode='lines',
                name='SNR',
                line=dict(color=COLOR_SNR, width=1.5),
            ),
            row=3, col=1
        )

        # Add threshold line
        snr_threshold = np.mean(snr_plot) * 0.75
        fig.add_hline(
            y=snr_threshold, row=3, col=1,
            line_dash="dash", line_color="gray",
            annotation_text=f"Threshold: {snr_threshold:.1f} dB"
        )

    # Update layout
    fig.update_layout(
        height=650,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (a.u.)", row=2, col=1)
    fig.update_yaxes(title_text="SNR (dB)", row=3, col=1)

    return fig


def create_pulse_plot(results):
    """Create the pulse boundaries and averaged pulse plot."""
    t = results['t']
    noisy = results['noisy']
    processed = results['processed']
    valleys = results['valleys']

    # Create subplot with boundaries and averaged pulse
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Signal with Pulse Boundaries", "Averaged Pulse Waveform"],
        horizontal_spacing=0.12
    )

    # Left plot: Signal with boundaries
    fig.add_trace(
        go.Scatter(
            x=t, y=noisy,
            mode='lines',
            name='Original (Noisy)',
            line=dict(color=COLOR_NOISY, width=1),
            legendgroup='left',
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=processed,
            mode='lines',
            name='LSTM Cleaned',
            line=dict(color=COLOR_CLEAN, width=1.2),
            legendgroup='left',
        ),
        row=1, col=1
    )

    # Add pulse boundary lines
    y_min = min(np.min(noisy), np.min(processed)) - 0.3
    y_max = max(np.max(noisy), np.max(processed)) + 0.3

    for i, v in enumerate(valleys):
        y0 = y_min - 0.1 if i % 2 == 0 else y_min
        y1 = y_max * 0.6

        fig.add_trace(
            go.Scatter(
                x=[t[v], t[v]],
                y=[y0, y1],
                mode='lines',
                line=dict(color='gray', width=0.8, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1, col=1
        )

    # Right plot: Averaged pulse waveform
    x_pulse = results['x_pulse']
    mean_noisy = results['mean_pulse_noisy']
    std_noisy = results['std_pulse_noisy']
    mean_processed = results['mean_pulse_processed']
    std_processed = results['std_pulse_processed']

    # Noisy pulse with error band
    fig.add_trace(
        go.Scatter(
            x=x_pulse, y=mean_noisy,
            mode='lines',
            name='Original Pulse',
            line=dict(color=COLOR_NOISY, width=2),
            legendgroup='right',
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_pulse, x_pulse[::-1]]),
            y=np.concatenate([mean_noisy - std_noisy, (mean_noisy + std_noisy)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 170, 28, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=1, col=2
    )

    # Processed pulse with error band
    fig.add_trace(
        go.Scatter(
            x=x_pulse, y=mean_processed,
            mode='lines',
            name='LSTM Pulse',
            line=dict(color=COLOR_CLEAN, width=2),
            legendgroup='right',
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_pulse, x_pulse[::-1]]),
            y=np.concatenate([mean_processed - std_processed, (mean_processed + std_processed)[::-1]]),
            fill='toself',
            fillcolor='rgba(64, 95, 193, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=1, col=2
    )

    # Add TTP vertical lines
    fig.add_vline(x=results['ttp_raw'], row=1, col=2, line_dash="dash",
                  line_color=COLOR_NOISY, line_width=1)
    fig.add_vline(x=results['ttp_lstm'], row=1, col=2, line_dash="dash",
                  line_color=COLOR_CLEAN, line_width=1)

    # Update layout with separate legends for each subplot
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (a.u.)", row=1, col=2)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    return fig


# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="LSTM Pulsation Tracing",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #405FC1;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .data-loaded {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">LSTM Pulsation Tracing</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Clean noisy optical signals (NIRS, PPG, DCS) to trace cardiac pulsations using a trained LSTM model.</p>',
        unsafe_allow_html=True
    )

    # Load model
    with st.spinner("Loading LSTM model..."):
        model, device = load_model()

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'signal_name' not in st.session_state:
        st.session_state.signal_name = None
    if 'signal' not in st.session_state:
        st.session_state.signal = None
    if 'fs' not in st.session_state:
        st.session_state.fs = None

    # Sidebar - Input Section
    with st.sidebar:
        st.header("Input Data")

        # Example datasets
        st.subheader("Example Datasets")
        col1, col2 = st.columns(2)

        with col1:
            use_fiber_shaking = st.button("Fiber Shaking", help="Load fiber shaking example from the paper")
        with col2:
            use_head_shaking = st.button("Head Shaking", help="Load head shaking example from the paper")

        st.divider()

        # File upload
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload signal file",
            type=['mat', 'csv', 'txt'],
            help="Supported formats: .mat, .csv, .txt"
        )

        # Sampling frequency input (blank by default)
        fs_input = st.number_input(
            "Sampling Frequency (Hz)",
            min_value=1.0,
            max_value=10000.0,
            value=None,
            step=1.0,
            placeholder="Enter sampling frequency",
            help="Original sampling frequency of your signal. Required for all uploaded files."
        )

        st.divider()

        # Delta OD conversion option
        st.subheader("Signal Type")
        convert_dod = st.checkbox(
            "Convert to Delta OD",
            value=True,
            help="Convert raw intensity signal to change in optical density: ΔOD = -ln(I/I₀). Enable for raw intensity signals (NIRS, DCS). Disable if your signal is already in ΔOD or other units."
        )

        st.divider()

        # Processing parameters (advanced)
        with st.expander("Advanced Parameters"):
            st.markdown("""
            **Preprocessing Parameters**

            These parameters control how the signal is normalized before LSTM processing.
            The goal is to scale the signal so the **averaged pulse** ranges roughly from **-1 to 1**.
            If your results look off, adjust these values and check the averaged pulse plot.
            """)

            amplitude_scale = st.slider(
                "Amplitude Scale",
                min_value=0.5,
                max_value=2.0,
                value=0.95,
                step=0.05,
                help="Scales the signal amplitude. Increase if pulses appear too small; decrease if clipped."
            )
            offset = st.slider(
                "DC Offset",
                min_value=-0.5,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Shifts the signal baseline. Adjust if the signal is not centered around zero."
            )

            st.markdown("---")
            st.markdown("**Window Parameters**")

            overlap_size = st.slider(
                "Window Overlap",
                min_value=0,
                max_value=2900,
                value=DEFAULT_OVERLAP_SIZE,
                step=100,
                help="Overlap between consecutive 3000-point LSTM windows. Higher overlap = smoother transitions but slower processing."
            )

        st.divider()

        # Process button
        process_btn = st.button("Process Signal", type="primary", use_container_width=True)

        st.divider()

        # How to Use section
        with st.expander("How to Use", expanded=False):
            st.markdown("""
            **Quick Start:**
            1. Click an example dataset button OR upload your own file
            2. Set the sampling frequency for uploaded files
            3. Choose whether to convert to Delta OD
            4. Click "Process Signal"
            5. View results and download outputs

            **Supported File Formats:**
            - `.mat`: MATLAB files (first array found will be used)
            - `.csv`: First column or 'signal' column
            - `.txt`: Whitespace-separated values

            **Tips:**
            - Signal should be a 1D time trace
            - Enable "Convert to Delta OD" for raw intensity signals
            - Use zoom/pan tools on plots to explore data
            - Check the averaged pulse ranges from -1 to 1

            **About the Metrics:**
            - **Pearson r**: Correlation between original and cleaned average pulse
            - **ΔTTP**: Time-to-peak difference in milliseconds
            """)

    # Handle example dataset buttons
    if use_fiber_shaking:
        example_path = os.path.join(EXAMPLE_DATA_DIR, "nirs_fiber_shaking.mat")
        data = loadmat(example_path)
        signal = np.array(data['signal']).flatten()
        fs = float(np.array(data['fs']).flatten()[0])
        st.session_state.signal = signal
        st.session_state.fs = fs
        st.session_state.signal_name = "nirs_fiber_shaking"
        st.session_state.results = None
        st.rerun()

    if use_head_shaking:
        example_path = os.path.join(EXAMPLE_DATA_DIR, "nirs_head_shaking.mat")
        data = loadmat(example_path)
        signal = np.array(data['signal']).flatten()
        fs = float(np.array(data['fs']).flatten()[0])
        st.session_state.signal = signal
        st.session_state.fs = fs
        st.session_state.signal_name = "nirs_head_shaking"
        st.session_state.results = None
        st.rerun()

    # Handle file upload
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_content = uploaded_file.read()

        if file_ext == 'mat':
            signal = load_mat_file(file_content, uploaded_file.name)
        elif file_ext == 'csv':
            signal = load_csv_file(file_content)
        elif file_ext == 'txt':
            signal = load_txt_file(file_content)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            signal = None

        if signal is not None:
            st.session_state.signal = signal
            st.session_state.signal_name = uploaded_file.name.rsplit('.', 1)[0]
            st.session_state.results = None
            # Don't set fs here - user must input it manually

    # Data loaded indicator
    if st.session_state.signal is not None:
        with st.sidebar:
            st.markdown(
                f'<div class="data-loaded">Data loaded: <strong>{st.session_state.signal_name}</strong><br>'
                f'Length: {len(st.session_state.signal):,} samples</div>',
                unsafe_allow_html=True
            )
            if st.session_state.fs is not None:
                st.markdown(
                    f'<div class="data-loaded">Sampling rate: <strong>{st.session_state.fs:.1f} Hz</strong></div>',
                    unsafe_allow_html=True
                )

    # Process signal
    if process_btn:
        if st.session_state.signal is None:
            st.error("Please load a dataset first (example or upload).")
        else:
            # Determine sampling frequency
            if st.session_state.fs is not None:
                # Example dataset - use stored fs
                fs_to_use = st.session_state.fs
            elif fs_input is not None:
                # User uploaded file with fs specified
                fs_to_use = fs_input
                st.session_state.fs = fs_input
            else:
                # No fs specified
                st.error("Please enter the sampling frequency for your uploaded data.")
                fs_to_use = None

            if fs_to_use is not None:
                with st.spinner("Processing signal with LSTM... This may take a moment."):
                    try:
                        results = process_signal(
                            model, device,
                            st.session_state.signal,
                            fs_to_use,
                            amplitude_scale=amplitude_scale,
                            offset=offset,
                            overlap_size=overlap_size,
                            convert_dod=convert_dod,
                        )
                        st.session_state.results = results
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error processing signal: {e}")
                        st.session_state.results = None

    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results

        # Metrics display
        st.subheader("Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Pearson Correlation (r)",
                value=f"{results['r_pearson']:.3f}",
                help="Shape similarity between original and cleaned pulse"
            )

        with col2:
            st.metric(
                label="ΔTTP",
                value=f"{results['delta_ttp']*1000:.1f} ms",
                help="Absolute time-to-peak difference"
            )

        with col3:
            st.metric(
                label="Detected Pulses",
                value=f"{len(results['valleys'])}",
                help="Number of pulse boundaries detected"
            )

        with col4:
            st.metric(
                label="Signal Duration",
                value=f"{results['t'][-1]:.1f} s",
                help="Total signal duration after resampling to 50 Hz"
            )

        st.divider()

        # Plot 1: Signal comparison with SNR
        st.subheader("Signal Comparison")
        fig1 = create_signal_plot(results)
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # Plot 2: Pulse boundaries and averaged pulse
        st.subheader("Pulse Analysis")
        fig2 = create_pulse_plot(results)
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Download section
        st.subheader("Download Results")
        col1, col2, col3, col4 = st.columns(4)

        signal_name = st.session_state.signal_name or "signal"

        with col1:
            cleaned_df = pd.DataFrame({
                'time_s': results['t'],
                'original': results['noisy'],
                'lstm_cleaned': results['processed']
            })
            csv_cleaned = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Signal (.csv)",
                data=csv_cleaned,
                file_name=f"{signal_name}_cleaned.csv",
                mime="text/csv"
            )

        with col2:
            valleys_df = pd.DataFrame({
                'valley_index': results['valleys'],
                'valley_time_s': results['t'][results['valleys']]
            })
            csv_valleys = valleys_df.to_csv(index=False)
            st.download_button(
                label="Download Pulse Boundaries (.csv)",
                data=csv_valleys,
                file_name=f"{signal_name}_boundaries.csv",
                mime="text/csv"
            )

        with col3:
            metrics = {
                'pearson_r': float(results['r_pearson']),
                'delta_ttp_ms': float(results['delta_ttp'] * 1000),
                'ttp_original_s': float(results['ttp_raw']),
                'ttp_lstm_s': float(results['ttp_lstm']),
                'num_pulses': int(len(results['valleys'])),
                'signal_duration_s': float(results['t'][-1]),
                'sampling_frequency_hz': float(results['fs']),
                'original_sampling_frequency_hz': float(results['fs_original']),
            }
            json_metrics = json.dumps(metrics, indent=2)
            st.download_button(
                label="Download Metrics (.json)",
                data=json_metrics,
                file_name=f"{signal_name}_metrics.json",
                mime="application/json"
            )

        with col4:
            pulse_df = pd.DataFrame({
                'time_s': results['x_pulse'],
                'mean_original': results['mean_pulse_noisy'],
                'std_original': results['std_pulse_noisy'],
                'mean_lstm': results['mean_pulse_processed'],
                'std_lstm': results['std_pulse_processed'],
            })
            csv_pulse = pulse_df.to_csv(index=False)
            st.download_button(
                label="Download Avg Pulse (.csv)",
                data=csv_pulse,
                file_name=f"{signal_name}_avg_pulse.csv",
                mime="text/csv"
            )

    else:
        # Show placeholder when no results
        st.info("Select an example dataset or upload your own data, then click 'Process Signal' to begin.")

        # Show brief info about the app
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### About This App

            This application uses a trained LSTM (Long Short-Term Memory) neural network to clean
            noisy optical signals for cardiac pulsation tracing.

            **Supported Signal Types:**
            - Near-Infrared Spectroscopy (NIRS)
            - Photoplethysmography (PPG)
            - Diffuse Correlation Spectroscopy (DCS)

            **Key Features:**
            - Automatic signal preprocessing and normalization
            - Sliding window LSTM inference
            - Pulse detection and segmentation
            - Quality metrics (Pearson correlation, TTP)
            - Interactive visualization
            """)

        with col2:
            st.markdown("""
            ### Processing Pipeline

            1. **Preprocessing**
               - Optionally convert intensity to delta optical density (dOD)
               - Resample to 50 Hz (training frequency)
               - High-pass filter and normalize to [-1, 1]

            2. **LSTM Inference**
               - Segment signal into 60-second windows
               - Process each window with trained LSTM
               - Recombine with overlap averaging

            3. **Pulse Analysis**
               - Detect pulse boundaries (valleys)
               - Segment individual pulses
               - Compute averaged waveforms
               - Calculate quality metrics
            """)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
    LSTM Pulsation Tracing | Based on research by Jingyi Wu et al. |
    <a href="mailto:jingyiwu@andrew.cmu.edu">Contact</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
