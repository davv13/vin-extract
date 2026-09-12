"""Measure VIN extraction latency (cold load + per-image)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.ocr_engine import get_ocr_engine, reset_ocr_engine
from app.services.vin_extractor import VinExtractor


def main() -> None:
    images = sorted((ROOT / "examples" / "images").glob("image_*"))[:3]
    if not images:
        print("No sample images found")
        return

    reset_ocr_engine()
    get_settings.cache_clear()
    settings = get_settings()

    t0 = time.perf_counter()
    engine = get_ocr_engine(settings)
    load_s = time.perf_counter() - t0
    print(f"OCR model load (once at startup): {load_s:.2f}s")

    extractor = VinExtractor(settings=settings, ocr=engine)
    times = []
    for path in images:
        data = path.read_bytes()
        t1 = time.perf_counter()
        result = extractor.extract_from_bytes(data)
        elapsed = time.perf_counter() - t1
        times.append(elapsed)
        print(
            f"{path.name:16} {elapsed:6.2f}s  "
            f"status={result.status:15} vin={result.vin} "
            f"conf={result.confidence_percent}%"
        )

    avg = sum(times) / len(times)
    print(f"\nPer-image average (models already loaded): {avg:.2f}s")
    print(f"Typical API request time after warmup: ~{avg:.1f}s")
    print(f"First request if models load on demand: ~{load_s + times[0]:.1f}s")


if __name__ == "__main__":
    main()
