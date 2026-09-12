"""Re-run specific hard sample images with fuller OCR settings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prefer orientation handling for rotated door stickers / windshield shots
os.environ.setdefault("PADDLE_USE_DOC_ORIENTATION", "true")
os.environ.setdefault("PADDLE_DET_MODEL", "PP-OCRv5_mobile_det")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.config import get_settings
from app.services.ocr_engine import get_ocr_engine, reset_ocr_engine
from app.services.vin_extractor import VinExtractor

NAMES = ["image_1.jpg", "image_9.png", "image_10.jpg", "image_11.jpeg"]


def main() -> None:
    get_settings.cache_clear()
    reset_ocr_engine()
    settings = get_settings()
    print(
        f"det={settings.paddle_det_model} doc_ori={settings.paddle_use_doc_orientation}",
        flush=True,
    )
    engine = get_ocr_engine(settings)
    extractor = VinExtractor(settings=settings, ocr=engine)
    results = []
    for name in NAMES:
        path = ROOT / "examples" / "images" / name
        print(f"=== {name} ===", flush=True)
        result = extractor.extract_from_bytes(path.read_bytes())
        row = {
            "file": name,
            "success": result.success,
            "status": result.status,
            "vin": result.vin,
            "confidence_percent": result.confidence_percent,
            "check_digit_valid": result.check_digit_valid,
            "candidates": result.candidates[:3],
            "message": result.message,
        }
        results.append(row)
        print(
            f"  {result.status} {result.vin} {result.confidence_percent}% "
            f"candidates={[c.get('vin') for c in result.candidates[:3]]}",
            flush=True,
        )
    out = ROOT / "examples" / "images" / "results_hard.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
