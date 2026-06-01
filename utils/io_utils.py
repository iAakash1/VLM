# utils/io_utils.py
"""
File I/O helpers for PlantDx.

Handles:
  - Saving uploaded images with UUID + timestamp + metadata JSON
  - JSON load / save wrappers
  - Image loading with error handling
"""

import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


# ── Image Uploads ────────────────────────────────────────────────────────────

def save_uploaded_image(
    image: Image.Image,
    original_name: str,
    save_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist an uploaded PIL image to ``save_dir``.

    Naming convention:
        <YYYY-MM-DD>_<uuid>.<ext>

    Alongside the image a companion ``.meta.json`` file is created with:
        uuid, original_name, saved_as, path, timestamp, + any caller metadata.

    Returns the metadata dict (same as what's written to .meta.json).
    """
    uid       = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    suffix    = Path(original_name).suffix.lower() or ".jpg"
    fname     = f"{timestamp[:10]}_{uid}{suffix}"

    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_path  = save_dir / fname
    image.save(str(img_path))

    record: Dict[str, Any] = {
        "uuid":          uid,
        "original_name": original_name,
        "saved_as":      fname,
        "path":          str(img_path.resolve()),
        "timestamp":     timestamp,
    }
    if metadata:
        record.update(metadata)

    meta_path = save_dir / (fname + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


def save_image_from_path(src_path: Path, save_dir: Path) -> Dict[str, Any]:
    """
    Copy an existing image file into save_dir under a UUID filename.
    Useful for moving PlantVillage test images into the uploads folder.
    """
    image = load_image(src_path)
    return save_uploaded_image(image, src_path.name, save_dir)


def load_image(path: Any) -> Image.Image:
    """Load a PIL image from a path string or Path object."""
    return Image.open(str(path)).convert("RGB")


# ── JSON Helpers ─────────────────────────────────────────────────────────────

def load_json(path: Any) -> Any:
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


# ── Dataset helpers ──────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(folder: Path) -> list:
    """Return all valid image paths inside a folder (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.suffix in VALID_EXTENSIONS
    )
