# scripts/03_inference.py
"""
PlantDx - Step 3: Batch Inference

Uses InferenceEngine to run the full pipeline (OOD + inference + confidence)
over one representative image per class and the full validation split.

Outputs:
    outputs/inference/results.json

Run from the project root (PlantDx/):
    python scripts/03_inference.py
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import Paths, load_config
from utils.io_utils import load_json, save_json, collect_images
from utils.model_utils import load_qwen_model, load_clip
from core.inference_engine import InferenceEngine


def run_per_class(engine, dataset_root, verbose=True):
    """One representative image per class - quick qualitative check."""
    results  = []
    folders  = sorted(f for f in dataset_root.iterdir() if f.is_dir())

    if verbose:
        print("\nPer-class inference ({} classes)".format(len(folders)))
        print("=" * 70)

    for folder in folders:
        images = collect_images(folder)
        if not images:
            continue

        image  = Image.open(str(images[0])).convert("RGB")
        result = engine.predict(image, check_ood=False)

        if verbose:
            print("Class      :", folder.name)
            print("Condition  :", result["condition"])
            print("Confidence : {:.1f} %".format(result["confidence"]))
            print("-" * 70)

        results.append({
            "image":      str(images[0]),
            "class_name": folder.name,
            **result,
        })

    return results


def run_val_set(engine, splits_file, max_samples):
    """Run inference on the validation split."""
    val_set = load_json(splits_file)["val"][:max_samples]
    print("\nVal-set inference ({} samples)...".format(len(val_set)))

    enriched = engine.predict_batch(val_set, check_ood=False)

    # Flatten for downstream evaluation
    results = []
    for rec in enriched:
        r = rec["result"]
        results.append({
            "image":      rec["image"],
            "class_name": rec["class_name"],
            "condition":  rec.get("condition", ""),
            "plant":      rec.get("plant", ""),
            "reference":  rec.get("caption", ""),
            "prediction": r.get("prediction", ""),
            "confidence": r.get("confidence", 0.0),
            "condition_pred": r.get("condition", ""),
            "severity_pred":  r.get("severity",  ""),
        })

    avg = sum(r["confidence"] for r in results) / max(len(results), 1)
    print("Val-set done. Mean Gen. Confidence Score: {:.1f} %".format(avg))
    return results


def main():
    cfg   = load_config()
    paths = Paths(cfg)
    paths.makedirs()

    if not paths.best_model.exists():
        print("ERROR: No model at", paths.best_model)
        print("Run: python scripts/02_train.py  first.")
        sys.exit(1)

    print("Loading model...")
    model, processor = load_qwen_model(str(paths.best_model), cfg["model"]["name"])

    print("Loading CLIP...")
    clip_model, clip_processor = load_clip(cfg["evaluation"]["clip_model"])

    engine = InferenceEngine(model, processor, clip_model, clip_processor, cfg)

    per_class = run_per_class(engine, paths.dataset_root)
    val_set   = run_val_set(engine, paths.splits_file,
                             cfg["evaluation"]["max_samples"])

    save_json({"per_class": per_class, "val_set": val_set}, paths.inference_out)

    print("\nSaved ->", paths.inference_out)
    print("Per-class :", len(per_class), "| Val-set :", len(val_set))


if __name__ == "__main__":
    main()
