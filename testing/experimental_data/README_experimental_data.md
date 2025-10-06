# Experimental Data

This folder contains experimental datasets used to evaluate the performance of the LSTM framework for pulsation tracing and motion artifact removal. These datasets were collected using multiple devices across NIRS, PPG, and DCS modalities, and were employed in different validation tasks in the paper.

---

## Datasets

### NIRS (Near-Infrared Spectroscopy)
- **`nirs_fiber_shaking.mat`**  
  - Device: **ISS Oxiplex**  
  - Used to demonstrate pulsation tracing under strong motion artifacts induced by **fiber shaking**.

- **`nirs_head_shaking.mat`**  
  - Device: **ISS Oxiplex**  
  - Used to demonstrate pulsation tracing under strong motion artifacts induced by **head shaking**.

- **`nirs_ecg.mat`**  
  - Device: **Artinis PortaLite**  
  - Simultaneous ECG measured with **Grapevine system (Ripple Neuro)**.  
  - Used to evaluate **HR extraction accuracy** by comparing LSTM-derived NIRS pulsations with ECG-derived ground truth.

### PPG (Photoplethysmography)
- **`ppg_subject1.mat`**, **`ppg_subject2.mat`**, **`ppg_subject3.mat`**, **`ppg_subject4.mat`**  
  - Device: **N-595 Nellcor finger pulse oximeter (Medtronic)**  
  - Used to demonstrate **pulse shape tracing, segmentation, and averaging**.

### DCS (Diffuse Correlation Spectroscopy)
- **`dcs_subject1_supine.mat`**  
  - Device: **Custom-built laboratory DCS system**  
  - Collected in **supine posture**.  
  - Used to demonstrate **pulse shape tracing, segmentation, and averaging**.

- **`dcs_subject2_valsava.mat`**  
  - Device: **Custom-built laboratory DCS system**  
  - Collected during **Valsalva maneuver**.  
  - Used to demonstrate **pulse shape tracing, segmentation, and averaging**.
  

---

## Notes
- Full DCS dataset can be found in [DOI: 10.1117/1.NPh.11.1.015003](https://doi.org/10.1117/1.NPh.11.1.015003).
- All datasets should be preprocessed and aligned for compatibility with the LSTM inference pipeline. See testing scripts for more details.  
- These datasets were selected to cover a **diverse range of noise/artifact conditions** and **different optical modalities**, showcasing the flexibility of the proposed framework. 

---

## Contact
Jingyi Wu — jingyiwu@andrew.cmu.edu