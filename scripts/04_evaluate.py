# scripts/04_evaluate.py
"""
PlantDx — Step 4: Comprehensive Evaluation

Reads   : outputs/inference/results.json  (from 03_inference.py)
Produces: outputs/evaluations/report.json

Metrics:
  Caption quality  → BLEU-1, BLEU-4, ROUGE-L
  Classification   → Accuracy, Macro F1, Precision, Recall, per-class F1
  Severity         → R², MAE
  CLIP similarity  → mean, median, std (reference vs. generated)

Run from the project root (PlantDx/):
    python scripts/04_evaluate.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import Paths, load_config
from utils.io_utils import load_json, save_json
from utils.model_utils import DEVICE
from utils.metrics import (
    parse_condition,
    parse_severity,
    build_full_report,
)


def compute_clip_scores(
    refs: list,
    hyps: list,
    clip_model,
    clip_processor,
) -> list:
    """Compute pairwise CLIP cosine similarity between reference and predicted captions."""
    scores = []
    for ref, hyp in zip(refs, hyps):
        inputs = clip_processor(
            text=[ref, hyp],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            feats = outputs.pooler_output

        feats  = feats / feats.norm(dim=-1, keepdim=True)
        cosine = float((feats[0] * feats[1]).sum().item())
        scores.append(cosine)
    return scores


def print_report(report: dict) -> None:
    cap = report["caption"]
    cls = report["classification"]
    sev = report["severity"]
    clip = report["clip"]

    print(f"\n{'='*55}")
    print(f"  PlantDx Evaluation Report  ({report['n_samples']} samples)")
    print(f"{'='*55}")
    print(f"\nCaption Quality")
    print(f"  BLEU-1   : {cap['bleu1']:.4f}")
    print(f"  BLEU-4   : {cap['bleu4']:.4f}")
    print(f"  ROUGE-L  : {cap['rougeL']:.4f}")
    print(f"\nClassification")
    print(f"  Accuracy : {cls['accuracy']:.4f}")
    print(f"  Macro F1 : {cls['macro_f1']:.4f}")
    print(f"  Precision: {cls['precision']:.4f}")
    print(f"  Recall   : {cls['recall']:.4f}")
    print(f"\nSeverity Regression")
    print(f"  R²       : {sev['r2']:.4f}")
    print(f"  MAE      : {sev['mae']:.4f}")
    print(f"\nCLIP Text Similarity")
    print(f"  Mean     : {clip['mean']:.4f}")
    print(f"  Median   : {clip['median']:.4f}")
    print(f"  Std      : {clip['std']:.4f}")
    print(f"\nPer-class F1")
    for cls_name, f1 in sorted(report["classification"]["per_class_f1"].items()):
        bar = "█" * int(f1 * 20)
        print(f"  {cls_name:<40} {f1:.4f}  {bar}")
    print(f"{'='*55}\n")


def main():
    cfg   = load_config()
    paths = Paths(cfg)
    paths.makedirs()

    if not paths.inference_out.exists():
        print(f"ERROR: Inference results not found at {paths.inference_out}")
        print("Run: python scripts/03_inference.py  first.")
        sys.exit(1)

    print("[1/3] Loading inference results...")
    data     = load_json(paths.inference_out)
    val_data = data["val_set"]
    n        = len(val_data)
    print(f"      {n} samples loaded")

    # ── Extract structured fields ─────────────────────────────
    refs             = [r["reference"]  for r in val_data]
    hyps             = [r["prediction"] for r in val_data]
    true_conditions  = [parse_condition(r["reference"])  for r in val_data]
    pred_conditions  = [parse_condition(r["prediction"]) for r in val_data]
    true_severities  = [parse_severity(r["reference"])   for r in val_data]
    pred_severities  = [parse_severity(r["prediction"])  for r in val_data]

    # ── CLIP similarity ───────────────────────────────────────
    print("[2/3] Computing CLIP text similarity...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model_name = cfg["evaluation"]["clip_model"]
    clip_model     = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    clip_scores = compute_clip_scores(refs, hyps, clip_model, clip_processor)

    # ── Build report ──────────────────────────────────────────
    print("[3/3] Computing all metrics...")
    report = build_full_report(
        refs=refs,
        hyps=hyps,
        true_conditions=true_conditions,
        pred_conditions=pred_conditions,
        true_severities=true_severities,
        pred_severities=pred_severities,
        clip_scores=clip_scores,
        n_samples=n,
    )

    save_json(report, paths.eval_report)
    print_report(report)
    print(f"Full report saved → {paths.eval_report}")


if __name__ == "__main__":
    main()