# core/run_manager.py
"""
RunManager — per-experiment tracking for PlantDx training runs.

Each call to RunManager() creates a new numbered folder:
    outputs/runs/run_001/
    outputs/runs/run_002/
    ...

Inside every run folder:
    config.yaml         ← frozen copy of training config
    logs/
        train.jsonl     ← one JSON line per optimizer step
        epochs.jsonl    ← one JSON line per epoch (train + val loss)
    plots/
        loss.png        ← train vs val loss curve (generated at end of training)
    metrics.json        ← final evaluation metrics if provided

Usage (in 02_train.py):
    run = RunManager(paths.runs_dir, cfg)

    # during training loop:
    run.log_step(global_step, loss=0.43, lr=2e-4, vram_gb=18.1)
    run.log_epoch(epoch=1, train_loss=0.38, val_loss=0.31)

    # after training:
    run.save_plots()
    run.save_metrics({"bleu1": 0.60, "rouge_l": 0.58, ...})
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Windows / servers
import matplotlib.pyplot as plt


class RunManager:
    """
    Creates and manages one numbered training-run directory.

    Args:
        runs_root : base directory for all runs (e.g. ``outputs/runs/``)
        config    : the full config dict — saved as a frozen snapshot
    """

    def __init__(self, runs_root: Path, config: Dict[str, Any]) -> None:
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)

        # Pick the next available run number
        existing  = sorted(runs_root.glob("run_*"))
        run_idx   = len(existing) + 1
        self.run_dir = runs_root / f"run_{run_idx:03d}"

        # Sub-directories
        self.logs_dir  = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        for d in [self.logs_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # File handles
        self._step_log  = self.logs_dir / "train.jsonl"
        self._epoch_log = self.logs_dir / "epochs.jsonl"

        # In-memory buffers for plotting
        self._step_losses:  list = []   # (global_step, loss)
        self._epoch_train:  list = []   # (epoch, train_loss)
        self._epoch_val:    list = []   # (epoch, val_loss)

        self._start_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # Save config snapshot
        self._save_config(config)
        print(f"[RunManager] Run directory: {self.run_dir}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_step(
        self,
        global_step: int,
        loss:        float,
        lr:          float,
        vram_gb:     float = 0.0,
    ) -> None:
        """Log one optimizer step."""
        record = {
            "step":    global_step,
            "loss":    round(loss, 6),
            "lr":      lr,
            "vram_gb": round(vram_gb, 2),
            "ts":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._append_jsonl(self._step_log, record)
        self._step_losses.append((global_step, loss))

    def log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
    ) -> None:
        """Log the summary for one complete epoch."""
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "ts":         datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._append_jsonl(self._epoch_log, record)
        self._epoch_train.append((epoch, train_loss))
        self._epoch_val.append((epoch, val_loss))

    # ── Plots ─────────────────────────────────────────────────────────────────

    def save_plots(self) -> None:
        """
        Generate and save training loss curves.

        Produces:
            plots/loss.png          ← train step loss + epoch train/val loss
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training — {self.run_dir.name}", fontsize=13, fontweight="bold")

        # ── Left: step-level train loss ───────────────────────────────────
        ax = axes[0]
        if self._step_losses:
            steps, losses = zip(*self._step_losses)
            ax.plot(steps, losses, color="#4ade80", linewidth=1.0, alpha=0.8, label="Train loss")
            # Smoothed overlay
            if len(losses) > 20:
                import numpy as np
                k = max(len(losses) // 30, 5)
                smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
                s_steps = steps[k - 1:]
                ax.plot(s_steps, smooth, color="#16a34a", linewidth=2.0, label=f"Smoothed (k={k})")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Loss")
            ax.set_title("Train Loss per Step")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No step data", ha="center", va="center", transform=ax.transAxes)

        _style_ax(ax)

        # ── Right: epoch train vs val loss ────────────────────────────────
        ax = axes[1]
        if self._epoch_train and self._epoch_val:
            t_epochs, t_losses = zip(*self._epoch_train)
            v_epochs, v_losses = zip(*self._epoch_val)
            ax.plot(t_epochs, t_losses, "o-", color="#4ade80",  linewidth=2, markersize=6, label="Train loss")
            ax.plot(v_epochs, v_losses, "s-", color="#f97316",  linewidth=2, markersize=6, label="Val loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Train vs Val Loss per Epoch")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No epoch data", ha="center", va="center", transform=ax.transAxes)

        _style_ax(ax)

        plt.tight_layout()
        out_path = self.plots_dir / "loss.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0d1a0f")
        plt.close(fig)
        print(f"[RunManager] Loss plot saved → {out_path}")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save final evaluation metrics alongside the run config."""
        out = self.run_dir / "metrics.json"
        with open(str(out), "w", encoding="utf-8") as f:
            json.dump({"run": self.run_dir.name, "metrics": metrics}, f, indent=2)
        print(f"[RunManager] Metrics saved → {out}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Write a frozen YAML snapshot of the config used for this run."""
        try:
            import yaml
            out = self.run_dir / "config.yaml"
            with open(str(out), "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            # Non-fatal: fall back to JSON
            out = self.run_dir / "config.json"
            with open(str(out), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, record: Dict) -> None:
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ── Matplotlib style helper ───────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor("#0d1a0f")
    ax.figure.set_facecolor("#0d1a0f")
    ax.tick_params(colors="#6b9e72")
    ax.xaxis.label.set_color("#6b9e72")
    ax.yaxis.label.set_color("#6b9e72")
    ax.title.set_color("#e2f5e5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3320")
    ax.grid(True, color="#1e3320", linewidth=0.5, alpha=0.7)
