# utils/metrics.py
"""
All evaluation metrics for PlantDx.

Covers:
  - Caption quality   : BLEU-1, BLEU-4, ROUGE-L
  - Classification    : Accuracy, Macro F1, Precision, Recall, per-class F1
  - Severity regression: R², MAE
  - Text parsing helpers for structured outputs
"""

import re
import nltk
import numpy as np
from typing import Any, Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
)
from sklearn.metrics import r2_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Severity numeric mapping ──────────────────────────────────────────────────

SEVERITY_SCALE: Dict[str, float] = {
    "none":               0.0,
    "mild":               1.0,
    "mild to moderate":   2.0,
    "moderate":           3.0,
    "moderate to severe": 4.0,
    "severe":             5.0,
}


def severity_to_score(text: str) -> float:
    """Convert a free-text severity label to a numeric 0–5 score."""
    low = text.lower().strip()
    for key, val in SEVERITY_SCALE.items():
        if key in low:
            return val
    return 3.0  # default: moderate


# ── Structured caption parsing ────────────────────────────────────────────────

def extract_field(text: str, field: str) -> str:
    """
    Pull a value from a structured 'Field: value.' line.
    Returns 'Unknown' if the field is not found.
    """
    pattern = rf"{re.escape(field)}:\s*(.+?)(?:\.|$|\n)"
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else "Unknown"


def parse_plant(text: str) -> str:
    return extract_field(text, "Plant")

def parse_condition(text: str) -> str:
    return extract_field(text, "Condition")

def parse_severity(text: str) -> str:
    return extract_field(text, "Severity")


# ── Caption quality ───────────────────────────────────────────────────────────

def compute_bleu(
    refs: List[List[str]],
    hyps: List[List[str]],
) -> Dict[str, float]:
    """
    Args:
        refs : list of tokenized reference strings  (one per sample)
        hyps : list of tokenized hypothesis strings (one per sample)
    Returns BLEU-1 and BLEU-4.
    """
    smooth = SmoothingFunction().method1
    refs_wrapped = [[r] for r in refs]   # corpus_bleu expects [[ref], ...]
    return {
        "bleu1": float(corpus_bleu(refs_wrapped, hyps,
                                    weights=(1, 0, 0, 0),
                                    smoothing_function=smooth)),
        "bleu4": float(corpus_bleu(refs_wrapped, hyps,
                                    weights=(.25, .25, .25, .25),
                                    smoothing_function=smooth)),
    }


def compute_rouge(
    refs: List[str],
    hyps: List[str],
) -> Dict[str, float]:
    """Returns mean ROUGE-L F1."""
    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return {"rougeL": float(np.mean(scores))}


# ── Classification ────────────────────────────────────────────────────────────

def compute_classification(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Returns accuracy, macro F1/precision/recall, and per-class F1.
    All inputs are condition label strings (not integers).
    """
    labels = sorted(set(y_true + y_pred))
    per_class = {
        cls: float(f1_score(y_true, y_pred, labels=[cls],
                             average="macro", zero_division=0))
        for cls in labels
    }
    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "macro_f1":     float(f1_score(y_true, y_pred,     average="macro", zero_division=0, labels=labels)),
        "precision":    float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "recall":       float(recall_score(y_true, y_pred,  average="macro", zero_division=0, labels=labels)),
        "per_class_f1": per_class,
    }


# ── Severity regression ───────────────────────────────────────────────────────

def compute_severity_regression(
    y_true_sev: List[str],
    y_pred_sev: List[str],
) -> Dict[str, float]:
    """
    Convert severity labels to numeric and compute R² + MAE.
    """
    y_true_num = [severity_to_score(s) for s in y_true_sev]
    y_pred_num = [severity_to_score(s) for s in y_pred_sev]
    return {
        "r2":  float(r2_score(y_true_num, y_pred_num)),
        "mae": float(mean_absolute_error(y_true_num, y_pred_num)),
    }


# ── CLIP similarity ───────────────────────────────────────────────────────────

def compute_clip_similarity(
    clip_scores: List[float],
) -> Dict[str, float]:
    arr = np.array(clip_scores, dtype=float)
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
    }


# ── Full report builder ───────────────────────────────────────────────────────

def build_full_report(
    refs:          List[str],
    hyps:          List[str],
    true_conditions: List[str],
    pred_conditions: List[str],
    true_severities: List[str],
    pred_severities: List[str],
    clip_scores:   List[float],
    n_samples:     int,
) -> Dict[str, Any]:
    """
    Assemble all metrics into a single report dict.
    This is what 04_evaluate.py calls and saves as report.json.
    """
    refs_tok = [r.lower().split() for r in refs]
    hyps_tok = [h.lower().split() for h in hyps]

    report: Dict[str, Any] = {
        "n_samples":    n_samples,
        "caption":      {**compute_bleu(refs_tok, hyps_tok),
                         **compute_rouge(refs, hyps)},
        "classification": compute_classification(true_conditions, pred_conditions),
        "severity":     compute_severity_regression(true_severities, pred_severities),
        "clip":         compute_clip_similarity(clip_scores),
    }
    return report
