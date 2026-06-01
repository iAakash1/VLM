# core/trainer.py
"""
Trainer - full training loop for PlantDx.

Extracted from 02_train.py so the script becomes a thin entry point.
All optimizer creation, scheduling, AMP, gradient accumulation,
validation, early stopping, checkpointing, and resume live here.

AMP note
--------
We use torch.autocast("cuda", dtype=torch.bfloat16) which IS automatic
mixed precision. GradScaler is included but disabled for bfloat16 because
bfloat16 has a wider dynamic range than float16 and does not overflow;
scaling gradients would be counter-productive. If you ever switch to
float16 (not recommended on this hardware), set scaler_enabled=True.

Resume
------
Every checkpoint saves a training_state.pt alongside the adapter weights:
    { epoch, global_step, best_val_loss, patience_counter }
Pass resume_from to __init__ to restore from any checkpoint folder.

Usage in 02_train.py:
    trainer = Trainer(model, processor, train_loader, val_loader, cfg, paths, run)
    trainer.train()

    # With resume:
    trainer = Trainer(..., resume_from="outputs/checkpoints/checkpoint-400")
    trainer.train()
"""

import torch
from pathlib import Path
from typing import Optional

from transformers import get_cosine_schedule_with_warmup

from core.run_manager import RunManager
from utils.model_utils import DEVICE
from utils.paths import Paths

TRAINING_STATE_FILE = "training_state.pt"


class Trainer:
    """
    Manages one complete training run of Qwen2.5-VL.

    Args:
        model        : PEFT-wrapped Qwen2.5-VL model on DEVICE
        processor    : AutoProcessor (saved alongside checkpoints)
        train_loader : DataLoader for the training split
        val_loader   : DataLoader for the validation split
        cfg          : full config dict from configs/config.yaml
        paths        : Paths instance
        run          : RunManager instance
        resume_from  : optional path to a checkpoint folder to resume from
    """

    def __init__(
        self,
        model,
        processor,
        train_loader,
        val_loader,
        cfg:          dict,
        paths:        Paths,
        run:          RunManager,
        resume_from:  Optional[str] = None,
    ) -> None:
        self.model        = model
        self.processor    = processor
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.paths        = paths
        self.run          = run

        tr = cfg["training"]
        self.epochs      = tr["epochs"]
        self.grad_accum  = tr["grad_accum"]
        self.save_steps  = tr["save_steps"]
        self.log_steps   = tr["log_steps"]
        self.patience    = tr["patience"]

        # Training-state variables (may be overwritten by resume)
        self.start_epoch      = 0
        self.global_step      = 0
        self.best_val_loss    = float("inf")
        self.patience_counter = 0

        # Derived step counts
        self.total_steps  = (len(train_loader) // self.grad_accum) * self.epochs
        self.warmup_steps = int(self.total_steps * tr["warmup_ratio"])

        # Optimizer + cosine scheduler with linear warmup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # AMP: autocast is always on (bfloat16). GradScaler is OFF for bfloat16;
        # enable only if switching to float16.
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        print("[Trainer] Steps  : {}  Warmup: {}".format(
            self.total_steps, self.warmup_steps))
        print("[Trainer] Device : {}".format(DEVICE))

        # Resume from checkpoint if requested
        if resume_from:
            self._restore_checkpoint(resume_from)

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> float:
        """Run the full training loop. Returns the best validation loss."""
        for epoch in range(self.start_epoch, self.epochs):
            avg_train_loss = self._train_epoch(epoch)
            val_loss       = self._validate(epoch)

            self.run.log_epoch(epoch + 1, train_loss=avg_train_loss, val_loss=val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                self._save_best()
                print("  Best model saved  (val_loss={:.4f})\n".format(val_loss))
            else:
                self.patience_counter += 1
                print("  No improvement. Patience {}/{}\n".format(
                    self.patience_counter, self.patience))
                if self.patience_counter >= self.patience:
                    print("  Early stopping triggered.")
                    break

        self.run.save_plots()

        print("=" * 60)
        print("Training complete | Best val loss: {:.4f}".format(self.best_val_loss))
        print("Best model  ->", self.paths.best_model)
        print("Run folder  ->", self.run.run_dir)
        print("Loss plot   ->", self.run.plots_dir / "loss.png")
        print("=" * 60)

        return self.best_val_loss

    # ── Private: training epoch ───────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """One pass over the training set. Returns mean train loss for the epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        running_loss = 0.0
        epoch_loss   = 0.0
        epoch_steps  = 0

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # AMP forward pass with bfloat16 autocast
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = self.model(**batch).loss / self.grad_accum

            # Backward through scaler (no-op for bfloat16; active for float16)
            self.scaler.scale(loss).backward()

            running_loss += loss.item() * self.grad_accum
            epoch_loss   += loss.item() * self.grad_accum
            epoch_steps  += 1

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.log_steps == 0:
                    avg     = running_loss / (self.log_steps * self.grad_accum)
                    lr_now  = self.scheduler.get_last_lr()[0]
                    vram_gb = torch.cuda.memory_allocated() / 1e9
                    vram_pk = torch.cuda.max_memory_allocated() / 1e9

                    self.run.log_step(
                        self.global_step, loss=avg, lr=lr_now, vram_gb=vram_gb
                    )
                    print(
                        "  Epoch {}/{} | Step {}/{} | Loss: {:.4f} | "
                        "LR: {:.2e} | VRAM: {:.1f}/{:.1f} GB".format(
                            epoch + 1, self.epochs,
                            self.global_step, self.total_steps,
                            avg, lr_now, vram_gb, vram_pk,
                        )
                    )
                    running_loss = 0.0

                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.global_step, epoch)

        return epoch_loss / max(epoch_steps, 1)

    # ── Private: validation ───────────────────────────────────────────────────

    def _validate(self, epoch: int) -> float:
        """Run the full validation set and return mean loss."""
        print("\n  Validation - epoch {}...".format(epoch + 1))
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    total += self.model(**batch).loss.item()

        val_loss = total / len(self.val_loader)
        print("  Epoch {} | Val loss: {:.4f}".format(epoch + 1, val_loss))
        return val_loss

    # ── Private: checkpointing ────────────────────────────────────────────────

    def _save_checkpoint(self, step: int, epoch: int) -> None:
        """Save adapter weights + training state for resume support."""
        ckpt = self.paths.checkpoint(step)
        self.model.save_pretrained(str(ckpt))
        self.processor.save_pretrained(str(ckpt))
        self._save_training_state(ckpt, epoch)
        print("  Checkpoint -> {}".format(ckpt))

    def _save_best(self) -> None:
        """Overwrite outputs/checkpoints/best/ with the current best model."""
        self.model.save_pretrained(str(self.paths.best_model))
        self.processor.save_pretrained(str(self.paths.best_model))
        self._save_training_state(self.paths.best_model, epoch=None)

    def _save_training_state(self, directory: Path, epoch) -> None:
        """Persist optimizer/scheduler/training state for resume."""
        state = {
            "epoch":           epoch,
            "global_step":     self.global_step,
            "best_val_loss":   self.best_val_loss,
            "patience_counter": self.patience_counter,
            "optimizer":       self.optimizer.state_dict(),
            "scheduler":       self.scheduler.state_dict(),
            "scaler":          self.scaler.state_dict(),
        }
        torch.save(state, str(Path(directory) / TRAINING_STATE_FILE))

    # ── Private: resume ───────────────────────────────────────────────────────

    def _restore_checkpoint(self, checkpoint_path: str) -> None:
        """
        Restore training state from a saved checkpoint folder.
        The adapter weights are already loaded by 02_train.py via PeftModel;
        this method only restores optimizer / scheduler / step counters.
        """
        state_file = Path(checkpoint_path) / TRAINING_STATE_FILE
        if not state_file.exists():
            print("[Trainer] WARNING: training_state.pt not found in {}.".format(
                checkpoint_path))
            print("[Trainer]          Resuming with fresh optimizer state.")
            return

        state = torch.load(str(state_file), map_location="cpu")

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])

        self.global_step      = state.get("global_step",      0)
        self.best_val_loss    = state.get("best_val_loss",     float("inf"))
        self.patience_counter = state.get("patience_counter", 0)

        resumed_epoch = state.get("epoch") or 0
        self.start_epoch = resumed_epoch + 1

        print("[Trainer] Resumed from: {}".format(checkpoint_path))
        print("[Trainer]   global_step      = {}".format(self.global_step))
        print("[Trainer]   best_val_loss    = {:.4f}".format(self.best_val_loss))
        print("[Trainer]   resuming epoch   = {}".format(self.start_epoch))
