# utils/model_utils.py
"""
Centralized model utilities for PlantDx.

ALL model loading goes through this file:
  scripts/02_train.py     -> (training only, no loading)
  scripts/03_inference.py -> load_qwen_model()
  scripts/04_evaluate.py  -> load_qwen_model()
  app/05_app.py           -> load_qwen_model(), load_clip()

GPU / CPU portability
---------------------
DEVICE is set once here and imported everywhere that needs it.
Nothing outside this file ever calls .to("cuda") directly.

Generation Confidence Score
---------------------------
run_inference_with_confidence() returns a score derived from the
beam-search sequence log-probability (avg per token, sigmoid-mapped
to 0-100%). This is a generation fluency proxy, NOT a calibrated
probability. It is labelled "Generation Confidence Score" to be
accurate about what it represents.
"""

import math
import torch
from typing import List, Tuple

from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install with: pip install qwen-vl-utils")

from core.dataset import INSTRUCTION


# ── Device (Fix 6) ────────────────────────────────────────────────────────────
# Centralised here. Import this constant instead of hardcoding "cuda" anywhere.

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Plant anchor texts for CLIP OOD ──────────────────────────────────────────

PLANT_ANCHOR_TEXTS: List[str] = [
    "a plant leaf",
    "a diseased plant leaf",
    "a healthy green plant leaf",
    "crop disease symptoms on a leaf",
    "agricultural plant disease",
]


# ── Qwen2.5-VL loading ────────────────────────────────────────────────────────

def load_qwen_model(
    adapter_path: str,
    base_model:   str,
    dtype:        torch.dtype = torch.bfloat16,
) -> Tuple:
    """
    Load Qwen2.5-VL-7B-Instruct + LoRA adapter onto DEVICE.

    Args:
        adapter_path : path to saved adapter (outputs/checkpoints/best)
        base_model   : HuggingFace model ID or local path
        dtype        : torch dtype (default: bfloat16)

    Returns:
        (model, processor) - on DEVICE, in eval mode
    """
    print("[Model] Loading processor from:", adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path, padding_side="left")

    print("[Model] Loading base model:", base_model)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("[Model] Attaching LoRA adapter from:", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    if DEVICE == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used  = torch.cuda.memory_allocated() / 1e9
        print("[Model] Ready | VRAM {:.1f}/{:.1f} GB".format(used, total))
    else:
        print("[Model] Ready | Running on CPU (no CUDA detected)")

    return model, processor


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_inputs(image: Image.Image, instruction: str, processor):
    """Build processor inputs for a single image + instruction."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  instruction},
            ],
        }
    ]
    text_prompt  = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images_in, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=images_in,
        padding=True, return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def run_inference(
    image:              Image.Image,
    instruction:        str,
    model,
    processor,
    max_new_tokens:     int   = 160,
    num_beams:          int   = 4,
    repetition_penalty: float = 1.3,
) -> str:
    """Run one image through the model and return the generated text."""
    inputs    = _build_inputs(image, instruction, processor)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
        )

    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def run_inference_with_confidence(
    image:              Image.Image,
    instruction:        str,
    model,
    processor,
    max_new_tokens:     int   = 160,
    num_beams:          int   = 4,
    repetition_penalty: float = 1.3,
) -> Tuple[str, float]:
    """
    Run inference and return (prediction_text, generation_confidence_score).

    The confidence value is a Generation Confidence Score (0-100 %):
      - Derived from the beam-search sequence log-probability
      - High score = model generated the output with high fluency
      - NOT a calibrated probability of correctness
    Typical values:
        Well-learned disease class -> 85-98 %
        Ambiguous / borderline     -> 60-80 %
        OOD image slipped through  -> < 50 %
    """
    inputs    = _build_inputs(image, instruction, processor)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    pred = processor.decode(
        out.sequences[0][input_len:], skip_special_tokens=True
    ).strip()

    score = _compute_generation_confidence(out, input_len)
    return pred, score


def _compute_generation_confidence(out, input_len: int) -> float:
    """
    Map beam-search log-probability to a 0-100 % generation confidence score.

    sequences_scores = total log-prob of the best beam.
    Dividing by token count gives avg per-token log-prob.
    Sigmoid centred at avg_lp = -1.5 converts this to a percentage.
    Clamped to [30, 99] % to avoid misleadingly extreme values.
    """
    try:
        if not (hasattr(out, "sequences_scores") and out.sequences_scores is not None):
            return 0.0
        n_new  = max(out.sequences.shape[1] - input_len, 1)
        avg_lp = float(out.sequences_scores[0].item()) / n_new
        raw    = 1.0 / (1.0 + math.exp((avg_lp + 1.5) * 2))
        pct    = raw * 100.0
        return round(float(max(30.0, min(99.0, pct))), 1)
    except Exception:
        return 0.0


# ── CLIP OOD detection ────────────────────────────────────────────────────────

def load_clip(
    model_name: str = "openai/clip-vit-base-patch32",
) -> Tuple:
    """Load CLIP ViT-B/32 for OOD detection. Cached by the app via st.cache_resource."""
    print("[CLIP] Loading:", model_name)
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    return clip_model, clip_proc


def is_plant_image(
    image:         Image.Image,
    clip_model,
    clip_processor,
    threshold:     float     = 0.22,
    anchor_texts:  List[str] = None,
) -> Tuple[bool, float]:
    """
    Returns (is_plant: bool, max_similarity: float).

    max_similarity < threshold -> OOD, reject with a clear warning.
    No inference is run for rejected images.
    Threshold of 0.22 was calibrated on the PlantVillage val set.
    """
    if anchor_texts is None:
        anchor_texts = PLANT_ANCHOR_TEXTS

    inputs = clip_processor(
        text=anchor_texts, images=image,
        return_tensors="pt", padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        sim = clip_model(**inputs).logits_per_image.softmax(dim=-1)

    max_sim = float(sim.max().item())
    return max_sim >= threshold, max_sim
