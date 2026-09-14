"""Filesystem boundaries and strict frame pairing."""

import json
from pathlib import Path
import cv2
import numpy as np


def image_paths(directory: Path) -> list[Path]:
    """List still images deterministically; videos must be extracted beforehand."""
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(
            f"No images in {directory}; add frames or run scripts/synthetic_demo.py"
        )
    return paths


def paired_paths(first: Path, second: Path) -> list[tuple[Path, Path]]:
    """Pair exact unique file stems; reject missing frames instead of guessing sync."""
    left, right = image_paths(first), image_paths(second)
    a, b = {p.stem: p for p in left}, {p.stem: p for p in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Duplicate frame stems")
    if a.keys() != b.keys():
        raise ValueError(f"Unpaired frame stems: {sorted(a.keys() ^ b.keys())[:10]}")
    return [(a[name], b[name]) for name in sorted(a)]


def read_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read BGR with Unicode-path support and no EXIF orientation adjustment."""
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if size is not None and (image.shape[1], image.shape[0]) != tuple(size):
        raise ValueError(f"Image size differs from calibration: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image, raising if encoding fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def write_json(path: Path, value: dict) -> None:
    """Serialize NumPy values as JSON; reject NaN and Infinity."""

    def convert(item: object) -> object:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False, default=convert, allow_nan=False
        ),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    """Load UTF-8 JSON, accepting an optional BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))
