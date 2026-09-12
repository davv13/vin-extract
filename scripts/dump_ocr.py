"""Dump raw OCR lines for sample images (debugging)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.image_preprocess import decode_image_bytes
from app.services.ocr_engine import get_ocr_engine, reset_ocr_engine

NAMES = [
    "image_5.jpg",
    "image_4.webp",
    "image_2.webp",
    "image_10.jpg",
    "image_11.jpeg",
    "image_9.png",
    "image_1.jpg",
]


def main() -> None:
    reset_ocr_engine()
    engine = get_ocr_engine(get_settings())
    for name in NAMES:
        path = ROOT / "examples" / "images" / name
        img = decode_image_bytes(path.read_bytes())
        lines = engine.read(img)
        print(f"==== {name}", flush=True)
        for line in lines:
            print(f"  {line.confidence:.3f} | {line.text}", flush=True)


if __name__ == "__main__":
    main()
