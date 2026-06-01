# utils/paths.py
"""
Centralized path registry for PlantDx.
Every script imports Paths() — paths are never hardcoded elsewhere.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"Config not found at '{config_path}'. "
            "Make sure you run scripts from the project root (PlantDx/)."
        )
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Paths:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        root:   Optional[str] = None,
    ) -> None:
        if config is None:
            config = load_config()
        self.config = config
        self.root   = Path(root) if root else Path.cwd()

        self.dataset_root = Path(config["paths"]["dataset_root"])

        d = config["paths"]["data"]
        self.raw_uploads  = self.root / d["raw_uploads"]
        self.processed    = self.root / d["processed"]
        self.captions_dir = self.root / d["captions"]
        self.splits_dir   = self.root / d["splits"]

        o = config["paths"]["outputs"]
        self.checkpoints_dir = self.root / o["checkpoints"]
        self.evaluations_dir = self.root / o["evaluations"]
        self.inference_dir   = self.root / o["inference"]
        self.runs_dir        = self.root / o["runs"]

        self.logs_dir            = self.root / "logs"
        self.prediction_logs_dir = self.logs_dir / "predictions"

        self.captions_file = self.captions_dir   / "captions.json"
        self.splits_file   = self.splits_dir     / "splits.json"
        self.best_model    = self.checkpoints_dir / "best"
        self.eval_report   = self.evaluations_dir / "report.json"
        self.inference_out = self.inference_dir   / "results.json"

    def checkpoint(self, step: int) -> Path:
        return self.checkpoints_dir / f"checkpoint-{step}"

    def makedirs(self) -> None:
        for d in [
            self.raw_uploads, self.processed, self.captions_dir,
            self.splits_dir, self.checkpoints_dir, self.evaluations_dir,
            self.inference_dir, self.runs_dir, self.prediction_logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[Paths] All directories ready under: {self.root}")

    def __repr__(self) -> str:
        return (
            f"Paths(\n"
            f"  dataset_root = {self.dataset_root}\n"
            f"  raw_uploads  = {self.raw_uploads}\n"
            f"  best_model   = {self.best_model}\n"
            f"  runs_dir     = {self.runs_dir}\n"
            f"  pred_logs    = {self.prediction_logs_dir}\n"
            f")"
        )
