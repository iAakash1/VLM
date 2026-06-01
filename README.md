# 🌿 PlantDx — Plant Disease Detection via Vision-Language Model

Fine-tuned **Qwen2.5-VL-7B-Instruct** on the PlantVillage dataset (15 disease classes, ~20,600 images across tomato, potato, and pepper bell). Produces structured diagnostic captions identifying plant species, disease condition, severity, causal pathogen, and visual symptoms.

---

## Project Structure

```
PlantDx/
├── configs/
│   └── config.yaml              ← ALL paths + hyperparameters
├── data/
│   ├── raw/uploads/             ← user-uploaded images (UUID + .meta.json)
│   ├── processed/
│   ├── captions/captions.json   ← full caption records
│   └── splits/splits.json       ← train/val split
├── outputs/
│   ├── checkpoints/
│   │   ├── checkpoint-200/      ← periodic saves (adapter + training_state.pt)
│   │   └── best/                ← best val loss — used by inference + app
│   ├── evaluations/report.json  ← full evaluation metrics
│   ├── inference/results.json   ← per-class + val-set predictions
│   └── runs/
│       ├── run_001/             ← frozen config, step logs, epoch logs, loss.png
│       └── run_002/
├── scripts/
│   ├── 01_prepare_dataset.py
│   ├── 02_train.py              ← thin entry point
│   ├── 03_inference.py
│   └── 04_evaluate.py
├── app/
│   └── 05_app.py                ← Streamlit demo
├── core/
│   ├── dataset.py               ← PlantVillageDataset, collate_fn, INSTRUCTION
│   ├── trainer.py               ← full training loop (AMP, resume, early stopping)
│   ├── run_manager.py           ← per-run folders, logs, loss plots
│   └── inference_engine.py      ← single orchestrator for all predictions
├── utils/
│   ├── paths.py                 ← centralized path registry
│   ├── io_utils.py              ← image save, JSON helpers
│   ├── metrics.py               ← BLEU, ROUGE, F1, R², severity parsing
│   └── model_utils.py           ← DEVICE, model loading, inference, CLIP OOD
├── logs/
│   └── predictions/             ← one JSON per inference call
├── requirements.txt
└── README.md
```

---

## Architecture

| File | Responsibility |
|------|---------------|
| `core/dataset.py` | `PlantVillageDataset`, `collate_fn`, `INSTRUCTION` (single source) |
| `core/trainer.py` | Optimizer, scheduler, AMP, grad accumulation, validation, checkpointing, resume |
| `core/run_manager.py` | Per-run folders, step/epoch logs, matplotlib loss curves |
| `core/inference_engine.py` | OOD → inference → confidence → field parsing (one `.predict()` call) |
| `utils/model_utils.py` | `DEVICE`, model loading, `run_inference_with_confidence`, CLIP OOD |
| `utils/paths.py` | Every path in the project — never hardcoded elsewhere |
| `app/05_app.py` | Streamlit UI — all inference goes through `InferenceEngine` |

---

## Setup

### 1. Place the project
Put the `PlantDx/` folder at:
```
C:\Users\Admin\Desktop\VLM\PlantDx\
```

### 2. Activate the virtual environment
```powershell
C:\Users\Admin\Desktop\VLM\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch (CUDA 12.1)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies
```powershell
pip install -r requirements.txt
```

### 5. Verify dataset path in config
Open `configs/config.yaml` and confirm:
```yaml
paths:
  dataset_root: "C:/Users/Admin/Desktop/VLM/PlantVillage"
```

---

## Run Order

All commands from the project root (`PlantDx/`), venv activated:

```powershell
# Step 1 — Build captions + train/val split
python scripts/01_prepare_dataset.py

# Step 2 — Fine-tune (~60-90 min on RTX 4500 Ada)
python scripts/02_train.py

# Step 2 (resume from checkpoint after interruption)
python scripts/02_train.py --resume outputs/checkpoints/checkpoint-400

# Step 3 — Batch inference on val set + per-class check
python scripts/03_inference.py

# Step 4 — Evaluate all metrics
python scripts/04_evaluate.py

# Step 5 — Launch Streamlit demo
streamlit run app/05_app.py
```

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Caption quality | BLEU-1, BLEU-4, ROUGE-L |
| Classification | Accuracy, Macro F1, Precision, Recall |
| Per-class | F1 per disease class |
| Severity | R² (regression), MAE |
| CLIP | Text-to-text cosine similarity |

Expected results (RTX 4500 Ada, 3 epochs, LoRA r=32):

| Metric | Value |
|--------|-------|
| BLEU-1 | 0.6047 |
| BLEU-4 | 0.4556 |
| ROUGE-L | 0.5869 |
| Accuracy | 98.5% |
| Macro F1 | 0.9827 |
| Severity R² | 0.9654 |
| Severity MAE | 0.025 |

---

## Run Tracking

Each training run creates `outputs/runs/run_NNN/` containing:
- `config.yaml` — frozen hyperparameter snapshot
- `logs/train.jsonl` — step-level loss, LR, VRAM
- `logs/epochs.jsonl` — epoch-level train + val loss
- `plots/loss.png` — train vs val loss curves

Compare runs by diffing their `config.yaml` files.

---

## OOD Rejection

Non-plant images are scored against five plant-anchor texts via **CLIP ViT-B/32**.  
If max softmax similarity < `0.22` (configurable in `config.yaml`), inference is blocked and a clear warning is shown. The rejection is still logged to `logs/predictions/`.

---

## Generation Confidence Score

The confidence value displayed in the app is a **Generation Confidence Score** — derived from the beam-search sequence log-probability (avg per token, sigmoid-mapped to 0–100%). It reflects generation fluency, not calibrated prediction probability.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `Qwen2VL` import error | Use `Qwen2_5_VLForConditionalGeneration` (with `_5_`) |
| Loss stuck at epoch 1 | Label masking uses `<|im_start|>assistant` token search in `input_ids` |
| `image_grid_thw` KeyError | Included in `collate_fn` and `__getitem__` |
| Streamlit JS errors | Pin `streamlit>=1.40.0,<1.45.0` |
| `tuple[str,float]` TypeError | Use `typing.Tuple` for Python 3.10 |
| CLIP safetensors error | `use_safetensors=True` in `CLIPModel.from_pretrained` |
| Flash-attention fails | Skip it — pipeline works without it |

---

## Hardware

Tested: Windows 10, Python 3.10.8, NVIDIA RTX 4500 Ada (24 GB VRAM), CUDA 12.1  
Expected VRAM: ~18.1 GB used / ~20.2 GB peak during training.
