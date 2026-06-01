# scripts/02_train.py
"""
PlantDx - Step 2: Fine-tune Qwen2.5-VL-7B-Instruct

Thin entry point. All training logic lives in core/trainer.py.

Run from the project root (PlantDx/):
    python scripts/02_train.py
    python scripts/02_train.py --resume outputs/checkpoints/checkpoint-400

Dataset : C:/Users/Admin/Desktop/VLM/PlantVillage
Model   : Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import Paths, load_config
from utils.io_utils import load_json
from core.dataset import PlantVillageDataset, collate_fn
from core.run_manager import RunManager
from core.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="PlantDx training script")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Path to a checkpoint folder to resume from "
             "(e.g. outputs/checkpoints/checkpoint-400)",
    )
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config()
    paths = Paths(cfg)
    paths.makedirs()

    tr  = cfg["training"]
    lo  = cfg["lora"]
    mod = cfg["model"]

    torch.manual_seed(tr["seed"])

    run = RunManager(paths.runs_dir, cfg)

    print("=" * 60)
    print("  PlantDx - Qwen2.5-VL-7B-Instruct Fine-tuning")
    print("  Run     :", run.run_dir.name)
    print("  Dataset : C:/Users/Admin/Desktop/VLM/PlantVillage")
    print("  Model   :", mod["name"])
    print("  LoRA    : r={} alpha={}".format(lo["r"], lo["alpha"]))
    print("  Batch   : {} x {} = {}".format(
        tr["batch_size"], tr["grad_accum"], tr["batch_size"] * tr["grad_accum"]))
    if args.resume:
        print("  Resume  :", args.resume)
    print("=" * 60)

    # ── Processor ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading processor...")
    if args.resume:
        processor = AutoProcessor.from_pretrained(args.resume, padding_side="left")
    else:
        processor = AutoProcessor.from_pretrained(mod["name"], padding_side="left")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/4] Loading base model in bfloat16...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mod["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.gradient_checkpointing_enable()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("      VRAM total: {:.1f} GB".format(vram))

    # ── LoRA: fresh or resumed ────────────────────────────────────────────────
    if args.resume:
        print("[3/4] Loading LoRA adapter from checkpoint:", args.resume)
        model = PeftModel.from_pretrained(base_model, args.resume, is_trainable=True)
    else:
        print("[3/4] Applying LoRA (r={}, alpha={})...".format(lo["r"], lo["alpha"]))
        lora_cfg = LoraConfig(
            r=lo["r"],
            lora_alpha=lo["alpha"],
            target_modules=lo["target_modules"],
            lora_dropout=lo["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("[4/4] Loading dataset...")
    if not paths.splits_file.exists():
        print("  ERROR: splits.json not found at", paths.splits_file)
        print("  Run:   python scripts/01_prepare_dataset.py  first.")
        sys.exit(1)

    splits   = load_json(paths.splits_file)
    train_ds = PlantVillageDataset(splits["train"], processor, tr["max_length"], augment=True)
    val_ds   = PlantVillageDataset(splits["val"],   processor, tr["max_length"], augment=False)

    sample     = train_ds[0]
    supervised = (sample["labels"] != -100).sum().item()
    print("\n  Pre-flight:")
    print("    Supervised tokens :", supervised, "/", sample["labels"].shape[0])
    print("    image_grid_thw    :", sample["image_grid_thw"].shape, " OK")
    print("    Train :", len(train_ds), " | Val :", len(val_ds))
    if supervised < 5:
        print("  FATAL: Label masking broken - exiting.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    trainer = Trainer(
        model, processor, train_loader, val_loader, cfg, paths, run,
        resume_from=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
