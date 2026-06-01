# core/inference_engine.py
"""
InferenceEngine - single orchestrator for the full prediction pipeline.

Centralises what was previously spread across:
  app/05_app.py         (OOD check + inference call + result rendering)
  utils/model_utils.py  (inference helpers)
  scripts/03_inference.py (batch loop)

Every surface that needs a prediction imports and calls this class.
model_utils.py still owns model loading and the raw inference calls;
InferenceEngine sits one level above and adds orchestration.

Usage:
    engine = InferenceEngine(model, processor, clip_model, clip_proc, cfg)
    result = engine.predict(image)          # dict, see below
    result = engine.predict(image, check_ood=False)  # skip OOD for known images

Return dict schema:
    {
        "status":     "ok" | "ood" | "error",
        "prediction": str,
        "confidence": float,        # Generation Confidence Score 0-100 %
        "ood_score":  float,        # CLIP max-similarity (0-1)
        "is_plant":   bool,
        "plant":      str,
        "condition":  str,
        "severity":   str,
        "error":      str | None,   # only present on status == "error"
    }
"""

from __future__ import annotations

from typing import Dict, Optional

from PIL import Image

from core.dataset import INSTRUCTION
from utils.model_utils import is_plant_image, run_inference_with_confidence
from utils.metrics import parse_condition, parse_plant, parse_severity


class InferenceEngine:
    """
    Orchestrates OOD detection, preprocessing, inference, confidence scoring,
    and field parsing into a single .predict() call.

    Args:
        model          : fine-tuned Qwen2.5-VL (PEFT-wrapped) on DEVICE
        processor      : AutoProcessor for Qwen2.5-VL
        clip_model     : CLIP ViT-B/32 for OOD detection (may be None)
        clip_processor : CLIPProcessor (may be None if clip_model is None)
        cfg            : full config dict from configs/config.yaml
    """

    def __init__(
        self,
        model,
        processor,
        clip_model,
        clip_processor,
        cfg: Dict,
    ) -> None:
        self.model          = model
        self.processor      = processor
        self.clip_model     = clip_model
        self.clip_processor = clip_processor

        eval_cfg           = cfg["evaluation"]
        self.ood_threshold = eval_cfg["ood_threshold"]

        inf_cfg                  = cfg["inference"]
        self.max_new_tokens      = inf_cfg["max_new_tokens"]
        self.num_beams           = inf_cfg["num_beams"]
        self.repetition_penalty  = inf_cfg["repetition_penalty"]

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image:        Image.Image,
        instruction:  Optional[str] = None,
        check_ood:    bool          = True,
    ) -> Dict:
        """
        Full pipeline: OOD check -> inference -> confidence -> field parsing.

        Args:
            image       : PIL RGB image
            instruction : custom prompt; falls back to core/dataset.INSTRUCTION
            check_ood   : set False when running on known PlantVillage images
                          (e.g., batch inference / evaluation)

        Returns:
            Result dict (see module docstring for schema).
        """
        if instruction is None:
            instruction = INSTRUCTION

        result: Dict = {
            "status":     "ok",
            "prediction": "",
            "confidence": 0.0,
            "ood_score":  0.0,
            "is_plant":   True,
            "plant":      "Unknown",
            "condition":  "Unknown",
            "severity":   "Unknown",
        }

        # ── Step 1: OOD check ─────────────────────────────────────────────
        if check_ood and self.clip_model is not None:
            is_plant, ood_score = is_plant_image(
                image,
                self.clip_model,
                self.clip_processor,
                threshold=self.ood_threshold,
            )
            result["is_plant"] = is_plant
            result["ood_score"] = round(float(ood_score), 4)

            if not is_plant:
                result["status"] = "ood"
                return result

        # ── Step 2: Inference + confidence ────────────────────────────────
        try:
            prediction, confidence = run_inference_with_confidence(
                image,
                instruction,
                self.model,
                self.processor,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
            )
        except Exception as exc:
            result["status"] = "error"
            result["error"]  = str(exc)
            return result

        result["prediction"] = prediction
        result["confidence"] = confidence

        # ── Step 3: Parse structured fields ───────────────────────────────
        result["plant"]     = parse_plant(prediction)
        result["condition"] = parse_condition(prediction)
        result["severity"]  = parse_severity(prediction)

        return result

    # ── Batch helper ──────────────────────────────────────────────────────────

    def predict_batch(
        self,
        records:     list,
        instruction: Optional[str] = None,
        check_ood:   bool          = False,
    ) -> list:
        """
        Run predict() over a list of record dicts (from splits.json).
        check_ood defaults to False for known PlantVillage images.

        Returns the same list with a "result" key added to each record.
        """
        results = []
        for i, rec in enumerate(records):
            try:
                img    = Image.open(rec["image"]).convert("RGB")
                result = self.predict(img, instruction=instruction, check_ood=check_ood)
            except Exception as exc:
                result = {"status": "error", "error": str(exc),
                          "prediction": "", "confidence": 0.0,
                          "plant": "", "condition": "", "severity": ""}
            results.append({**rec, "result": result})

            if (i + 1) % 25 == 0:
                done     = i + 1
                avg_conf = sum(
                    r["result"].get("confidence", 0) for r in results
                ) / done
                print("  [{}/{}]  avg Gen. Confidence Score: {:.1f} %".format(
                    done, len(records), avg_conf))

        return results
