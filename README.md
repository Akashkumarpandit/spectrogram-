# SpectraGuard: AASIST Audio Anti-Spoofing & Deepfake Detection
### Combined ASVspoof 2019 & ASVspoof 2021 Pipeline

A complete, state-of-the-art voice anti-spoofing and audio deepfake detection system based on **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks). This project supports combined training and evaluation across **ASVspoof 2019** (LA, PA, DF) and **ASVspoof 2021** (LA, DF) protocols.

---

## 🚀 Key Features

- **Model Architecture**:
  - **AASIST** & **AASIST-L**: Raw waveform SincNet filterbanks + Residual spectro-temporal encoders + Graph Attention Networks (GAT) with spectral & temporal graph fusion.
  - Support for custom feature dimensions, hidden dims, and attention heads.
- **Loss Functions**:
  - **AM-Softmax** (Angular Margin Softmax with configurable margin $m$ and scale $s$) for highly discriminative deepfake feature separation.
  - Weighted Cross-Entropy and Focal Loss support.
- **Dataset Support**:
  - Seamless unified protocol parser for **ASVspoof 2019** (train/dev/eval) and **ASVspoof 2021** (eval with trial keys).
  - Multi-dataset protocol concatenation and balancing.
  - Robust audio preprocessing: auto-resampling to 16 kHz, repeat-padding, and random/center slicing (default 64,600 samples / ~4 seconds).
- **Evaluation & Benchmarking**:
  - **Equal Error Rate (EER)** calculation using ROC / optimal threshold search.
  - **minimum tandem Detection Cost Function (min t-DCF)** metric support matching official ASVspoof evaluation criteria.
- **Interactive Web UI**:
  - Sleek modern dark-mode interface with real-time microphone recording and audio file drag-and-drop.
  - Live WebAudio waveform visualizer, deepfake probability gauge meter, spoof classification confidence, and spectro-temporal breakdown.
- **Synthetic Test Generator**:
  - Built-in generator to synthesize sample audio files and protocol manifests for immediate pipeline verification without needing 50GB of raw datasets downloaded first.

---

## 📁 Project Structure

```
asvspoof-deepfake-detection/
├── configs/
│   ├── config.yaml                    # Master training & dataset configuration
│   └── aasist.yaml                    # AASIST model hyper-parameters
├── src/
│   ├── models/
│   │   ├── sincnet.py                 # SincNet learnable bandpass filterbank
│   │   ├── graph_modules.py           # Graph Attention (GAT) & Heterogeneous pooling
│   │   ├── aasist.py                  # AASIST and AASIST-L model definition
│   │   └── loss.py                    # AM-Softmax and loss criteria
│   ├── datasets/
│   │   ├── protocol_parser.py         # ASVspoof 2019/2021 protocol parser
│   │   ├── dataset.py                 # PyTorch Dataset for combined datasets
│   │   └── augmentations.py           # RawBoost & audio augmentations
│   ├── utils/
│   │   ├── metrics.py                 # EER and min t-DCF calculation
│   │   └── checkpoint.py              # Checkpoint save/load utilities
│   ├── train.py                       # Full training and validation pipeline
│   ├── evaluate.py                    # Evaluation script for test protocols
│   └── predict.py                     # Single/batch inference CLI
├── web/
│   ├── app.py                         # FastAPI backend with prediction endpoint
│   └── static/
│       ├── index.html                 # Modern glassmorphism web interface
│       ├── style.css                  # UI styling & animations
│       └── app.js                     # Audio recording, waveform, and API calls
├── synthetic_data_generator.py        # Generates sample data for immediate testing
└── requirements.txt                   # Dependencies
```

---

## 🛠️ Quick Start

### 1. Installation

```bash
cd asvspoof-deepfake-detection
pip install -r requirements.txt
```

*(On Apple Silicon / macOS, PyTorch will automatically utilize `mps` if available; otherwise `cpu` or `cuda` on NVIDIA GPUs).*

### 2. Verify with Synthetic Data

If you haven't yet downloaded the complete ASVspoof datasets, generate a mini mock dataset in seconds:

```bash
python synthetic_data_generator.py --output_dir ./sample_data --num_samples 60
```

### 3. Training

Train on combined ASVspoof 2019 and 2021 protocols:

```bash
python -m src.train --config configs/config.yaml
```

### 4. Running the Interactive Web Demo

Launch the web application:

```bash
python -m web.app --port 8000
```
Then open `http://localhost:8000` in your browser.
