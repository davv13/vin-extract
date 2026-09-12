"""Run VIN extraction on all images in examples/images/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.ocr_engine import get_ocr_engine
from app.services.vin_extractor import VinExtractor

IMAGE_DIR = ROOT / "examples" / "images"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    images = sorted(
        (p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.stat().st_size,
    )
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    settings = get_settings()
    print(f"Loading OCR ({settings.ocr_backend})…")
    engine = get_ocr_engine(settings)
    extractor = VinExtractor(settings=settings, ocr=engine)

    results = []
    print(f"Processing {len(images)} images from {IMAGE_DIR}\n", flush=True)

    for path in images:
        print(f"=== {path.name} ===", flush=True)
        data = path.read_bytes()
        result = extractor.extract_from_bytes(data)
        row = {
            "file": path.name,
            "success": result.success,
            "status": result.status,
            "vin": result.vin,
            "confidence_percent": result.confidence_percent,
            "check_digit_valid": result.check_digit_valid,
            "message": result.message,
        }
        results.append(row)
        print(
            f"  status={result.status}  vin={result.vin}  "
            f"confidence={result.confidence_percent}%  "
            f"check_digit={result.check_digit_valid}",
            flush=True,
        )
        print(f"  {result.message}\n", flush=True)

    out_path = IMAGE_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved summary → {out_path}")

    accepted = sum(1 for r in results if r["success"])
    low = sum(1 for r in results if r["status"] == "low_confidence")
    missing = sum(1 for r in results if r["status"] == "not_found")
    print(f"\nSummary: {accepted} accepted, {low} low_confidence, {missing} not_found")


if __name__ == "__main__":
    main()
