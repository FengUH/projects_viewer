# 🔗 Live Demo

This repository contains a **neuroscience research demo app** built with **Streamlit**, showcasing an end-to-end workflow from **single-subject EEG time-series analysis** to **group-level network statistics** and **ML-based disease classification**.  

👉  Click to open the live Streamlit app: [https://feng-eeg-viewer.streamlit.app](https://fenguh-projects-viewer-projects-viewer-vwnh2n.streamlit.app/)

(Public link — no login required)
 
---


## 🧩 Main Functional Modules

### 1️⃣ Single-Subject EEG Time-Series Analysis

- Mock loading of a **single-subject EEG recording** (multi-channel).
- Interactive controls for:
  - Time window length & start time  
  - Number of channels to display
- Visualization of:
  - **Raw multi-channel EEG traces**
  - Optional **moving-average smoothing** with adjustable kernel size
  - Optional **window-level connectivity matrix** (Pearson correlation)
  - Optional **dynamic functional connectivity (dFNC)**:
    - Sliding-window correlation
    - **k-means clustering** to identify recurring brain states
    - Panel-style figure summarizing windows + state time course

---

### 2️⃣ Group-Level Statistics & Regression

(Accessible via **Group analysis → Statistics**)

- Synthetic **group-level connectivity matrices** for:
  - **Depression**
  - **Alzheimer’s disease (AD)**
- Side-by-side comparison of functional connectivity patterns between groups.
- Graph-theoretic summaries (using synthetic data):
  - **Node degree** and **node strength**
  - Visualized via two-group **boxplots** (Deppression vs AD)
- Simple **linear regression** from network metrics to mock clinical scores:
  - Scatter plots + fitted lines
  - Regression equation & R² displayed in the figure

---

### 3️⃣ ML-Based Classification

(Accessible via **Group analysis → ML classification**)

- Sidebar **ML workflow**:
  - Train a mock CNN on group-level connectivity features  
  - Or load a mock pre-trained model
  - Load a demo subject and apply the model
- Main tab visualizes:
  - **Connectivity → CNN → Diagnosis** conceptual diagram
  - **CNN output** as class probability bars (Depression vs AD)
  - **Model performance summary** :
  - **Subject-level prediction card** with highlighted final diagnosis

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **Matplotlib** 

---
