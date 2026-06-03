# 🌿 PlantDx: Plant Disease Detection using Vision-Language Models

PlantDx is a vision-language based plant disease detection system built by fine-tuning **Qwen2.5-VL-7B-Instruct** on the PlantVillage dataset.

The model generates structured diagnostic captions that identify:

* 🌱 Plant species
* 🦠 Disease condition
* 📈 Severity level
* 🔬 Causal pathogen
* 👀 Visible symptoms

The project currently supports **15 disease classes** across tomato, potato, and bell pepper crops using ~20,600 images.

---

# Features

* Fine-tuned Vision-Language Model using Qwen2.5-VL
* Structured disease caption generation
* Out-of-Distribution (OOD) rejection using CLIP
* Confidence scoring for predictions
* Full training pipeline with checkpointing
* Streamlit-based inference application
* Detailed experiment tracking and evaluation

---

# Project Structure

```text
PlantDx/
├── app/                     # Streamlit application
├── configs/                 # Configuration files
├── core/                    # Training and inference logic
├── scripts/                 # Training / evaluation scripts
├── utils/                   # Utility modules
├── data/                    # Dataset + generated files
├── outputs/                 # Checkpoints and results
├── logs/                    # Prediction logs
├── requirements.txt
└── README.md
```

---

# Architecture Overview

| Module                     | Purpose                                  |
| -------------------------- | ---------------------------------------- |
| `core/dataset.py`          | Dataset loading and preprocessing        |
| `core/trainer.py`          | Training loop, optimization, checkpoints |
| `core/inference_engine.py` | Unified prediction pipeline              |
| `core/run_manager.py`      | Run tracking and logging                 |
| `utils/model_utils.py`     | Model loading and helper functions       |
| `utils/metrics.py`         | Evaluation metrics                       |
| `utils/paths.py`           | Centralized path management              |
| `app/05_app.py`            | Streamlit frontend                       |

---

# Dataset

Dataset used: **PlantVillage**

Supported crops:

* Tomato
* Potato
* Bell Pepper

Total classes: **15 disease categories**

Dataset configuration is controlled through:

```yaml
configs/config.yaml
```

Example:

```yaml
paths:
  dataset_root: "/path/to/PlantVillage"
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd PlantDx
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install PyTorch

CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training Pipeline

Run commands from project root.

## Step 1: Prepare Dataset

```bash
python scripts/01_prepare_dataset.py
```

## Step 2: Train Model

```bash
python scripts/02_train.py
```

Resume training:

```bash
python scripts/02_train.py --resume outputs/checkpoints/checkpoint-400
```

## Step 3: Run Inference

```bash
python scripts/03_inference.py
```

## Step 4: Evaluate Model

```bash
python scripts/04_evaluate.py
```

## Step 5: Launch Application

```bash
streamlit run app/05_app.py
```

---

# Evaluation Metrics

| Category            | Metrics                               |
| ------------------- | ------------------------------------- |
| Caption Quality     | BLEU-1, BLEU-4, ROUGE-L               |
| Classification      | Accuracy, Precision, Recall, Macro F1 |
| Severity Estimation | MAE, R²                               |
| Similarity          | CLIP cosine similarity                |

Expected performance:

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 98.5%  |
| Macro F1    | 0.9827 |
| BLEU-1      | 0.6047 |
| BLEU-4      | 0.4556 |
| ROUGE-L     | 0.5869 |
| Severity R² | 0.9654 |

---

# Run Tracking

Each run automatically generates:

```text
outputs/runs/run_xxx/
```

Includes:

* Frozen config snapshot
* Training logs
* Validation logs
* Loss curves
* Hyperparameter history

---

# OOD Rejection

PlantDx uses **CLIP ViT-B/32** to reject unrelated images.

If similarity score is below threshold:

```text
0.22
```

prediction is blocked and logged.

---

# Confidence Score

Displayed confidence is generated from:

* Beam-search log probabilities
* Token-level confidence aggregation
* Sigmoid normalization

This score represents **generation confidence**, not calibrated probability.

---

# Common Issues

| Problem                  | Solution                        |
| ------------------------ | ------------------------------- |
| Import errors            | Use correct Qwen2.5 class names |
| Training stuck           | Verify label masking            |
| Streamlit issues         | Use supported version           |
| CLIP loading errors      | Enable safetensors              |
| Flash attention failures | Disable it                      |

---

# Hardware Tested

* Windows 10
* Python 3.10.8
* NVIDIA RTX 4500 Ada (24GB VRAM)
* CUDA 12.1

Approximate training memory:

* Average usage: ~18 GB VRAM
* Peak usage: ~20 GB VRAM

---

# Future Improvements

* More crop categories
* Mobile deployment
* Multilingual outputs
* Larger datasets
* Calibration improvements

---
