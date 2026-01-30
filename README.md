# LSTM Pulsation Tracing - Web Demo

A Streamlit web application for cleaning noisy optical signals (NIRS, PPG, DCS) to trace cardiac pulsations using a trained LSTM model. https://lstm-pulsation-demo.streamlit.app/

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This demo accompanies the paper:

> **"A Synthetic-Data-Driven LSTM Framework for Tracing Cardiac Pulsation in Optical Signals"**
> Wu, J.\*, Bai, S.\*, et al. *Biomedical Optics Express*, 2025.
> [https://doi.org/10.1364/BOE.574286](https://doi.org/10.1364/BOE.574286)

The application allows you to:
- Upload your own optical signal data (.mat, .csv, or .txt files)
- Process signals with a pre-trained LSTM model
- Visualize original vs. cleaned signals with interactive plots
- Analyze detected pulses and averaged waveforms
- Download processed data and quality metrics

## Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/jingyiwu-biophotonics/lstm-pulsation-demo.git
cd lstm-pulsation-demo
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

To deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your forked repository
4. Set the main file path to `app.py`
5. Click "Deploy"

## Usage

### Input Data

The app accepts three file formats:

| Format | Description | Requirements |
|--------|-------------|--------------|
| `.mat` | MATLAB file | Should contain `signal` and `fs` (sampling frequency) fields |
| `.csv` | CSV file | First column or column named `signal` |
| `.txt` | Text file | Whitespace-separated values, one sample per line |

For `.csv` and `.txt` files, you must specify the sampling frequency manually.

### Example Datasets

Two example datasets from the paper are included:
- **Fiber Shaking**: NIRS signal with fiber movement artifacts
- **Head Shaking**: NIRS signal with head movement artifacts

Click the corresponding button in the sidebar to load these examples.

### Processing Pipeline

1. **Preprocessing**
   - Convert intensity to delta optical density (dOD)
   - Resample to 50 Hz (matches training data)
   - High-pass filter (0.5 Hz cutoff)
   - Normalize amplitude to [-1, 1] range

2. **LSTM Inference**
   - Segment signal into 60-second overlapping windows
   - Process each segment with the trained LSTM
   - Recombine segments with overlap averaging

3. **Pulse Analysis**
   - Detect pulse boundaries (valleys) on cleaned signal
   - Segment individual pulses
   - Filter outliers using z-score
   - Compute averaged waveforms with standard deviation

### Output Metrics

- **Pearson r**: Correlation between original and LSTM-cleaned averaged pulses (shape similarity)
- **ΔTTP**: Absolute difference in time-to-peak between original and cleaned pulses

### Downloads

The app provides download buttons for:
- Cleaned signal (.csv)
- Pulse boundaries (.csv)
- Quality metrics (.json)
- Averaged pulse waveforms (.csv)

## Project Structure

```
lstm-pulsation-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── models/
│   └── lstm_full_dataset.pt    # Pre-trained LSTM weights
├── utils/
│   ├── __init__.py
│   ├── lstm_detection_model.py # Model architecture
│   ├── signal_processing.py    # Preprocessing functions
│   ├── detection_utils.py      # Pulse detection utilities
│   ├── windowing.py           # Signal segmentation
│   └── inference_helpers.py    # LSTM inference helpers
└── data/
    ├── nirs_fiber_shaking.mat  # Example dataset
    └── nirs_head_shaking.mat   # Example dataset
```

## Advanced Parameters

In the sidebar under "Advanced Parameters":

- **Amplitude Scale** (0.5-2.0): Controls vertical scaling of normalized signal. Adjust if pulses appear too small or clipped.
- **DC Offset** (-0.5 to 0.5): Shifts signal center. Useful if signal baseline is offset from zero.

## Tips for Best Results

1. **Signal Quality**: The LSTM works best on signals that contain visible (even if noisy) cardiac pulsations
2. **Preprocessing**: If results look poor, try adjusting the amplitude scale parameter
3. **Sampling Rate**: Ensure you input the correct sampling frequency for your data
4. **Signal Format**: For NIRS data, intensity signals are automatically converted to dOD

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

Key dependencies:
- streamlit >= 1.28.0
- torch >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- plotly >= 5.18.0

## Citation

If you use this application or the underlying model, please cite:

```bibtex
@article{wu2025lstm_pulsation_tracing,
  title   = {A Synthetic-Data-Driven LSTM Framework for Tracing Cardiac Pulsation in Optical Signals},
  author  = {Wu, Jingyi and Bai, Shaojie and Ozkaya, Zeynep and Patel, Justin A. and Skog, Emily and Ruesch, Alexander and Smith, Matthew A. and Kainerstorfer, Jana M.},
  journal = {Biomedical Optics Express},
  year    = {2025},
  doi     = {10.1364/BOE.574286}
}
```

## License

- **Code**: [MIT License](LICENSE)
- **Pre-trained Model & Data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Contact

**Jingyi Wu** — [jingyiwu@andrew.cmu.edu](mailto:jingyiwu@andrew.cmu.edu)

---

*This is a demo application for research purposes. For the full research repository including training code and additional datasets, see the [main repository](https://github.com/jingyiwu-biophotonics/LSTM-Pulsation-Tracing).*
