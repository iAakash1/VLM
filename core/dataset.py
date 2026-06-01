# core/dataset.py
"""
PlantVillage dataset and DataLoader helpers.

Imported by:
  scripts/02_train.py     (training)
  scripts/03_inference.py (batch inference)
  scripts/04_evaluate.py  (evaluation)
  app/05_app.py           (INSTRUCTION constant only)

Nothing here depends on training config. It only needs a processor
and a list of record dicts (from data/splits/splits.json).
"""

import torch
from typing import Dict, List

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen_vl_utils not found. "
        "Install with: pip install qwen-vl-utils"
    )


# ── Shared instruction ────────────────────────────────────────────────────────
# Single source of truth: change here and it propagates to train / infer / eval / app.

INSTRUCTION = (
    "Analyze this plant leaf image and provide a structured diagnosis. "
    "State: Plant name, Disease condition, Severity level, "
    "Causal pathogen, and visible symptoms observed on the leaf."
)


# ── Augmentation pipeline ─────────────────────────────────────────────────────

AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    Converts (image_path, caption) record dicts into Qwen2.5-VL chat-format tensors.

    Label masking
    -------------
    We locate the <|im_start|>assistant token sequence directly inside
    input_ids.  Prompt-length-only calculation is unreliable because the
    processor inserts a variable number of image tokens that the tokenizer
    alone does not know about.

    Args:
        records    : list of dicts — must contain keys ``image`` (abs path)
                     and ``caption`` (str).
        processor  : AutoProcessor for Qwen2.5-VL.
        max_length : combined prompt + caption token budget.
        augment    : apply random image augmentations when True.
    """

    def __init__(
        self,
        records:    List[Dict],
        processor,
        max_length: int  = 512,
        augment:    bool = False,
    ) -> None:
        self.records    = records
        self.processor  = processor
        self.max_length = max_length
        self.augment    = augment

        # Encode the assistant marker once — reused for every sample
        self._asst_ids: List[int] = processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )

    def __len__(self) -> int:
        return len(self.records)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_assistant_start(self, ids: torch.Tensor) -> int:
        """Index of the first token after the assistant marker (or 0 as fallback)."""
        marker   = self._asst_ids
        n        = len(marker)
        ids_list = ids.tolist()
        for i in range(len(ids_list) - n):
            if ids_list[i : i + n] == marker:
                return i + n
        return 0

    def _augment_image(self, image: Image.Image) -> Image.Image:
        tensor = T.ToTensor()(image)
        tensor = AUGMENT(tensor)
        return T.ToPILImage()(tensor)

    # ── Core ──────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec     = self.records[idx]
        image   = Image.open(rec["image"]).convert("RGB")
        caption = rec["caption"]

        if self.augment:
            image = self._augment_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]

        text_prompt  = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        images_in, _ = process_vision_info(messages)

        enc = self.processor(
            text=[text_prompt],
            images=images_in,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values   = enc["pixel_values"].squeeze(0)
        image_grid_thw = enc["image_grid_thw"].squeeze(0)  # required by Qwen2.5-VL

        # Mask prompt + padding tokens from the loss
        labels = input_ids.clone()
        asst_start = self._find_assistant_start(input_ids)
        labels[:asst_start] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels":         labels,
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack a list of dataset dicts into one batched dict of tensors."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
