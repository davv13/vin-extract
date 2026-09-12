"""Smoke-test OCR + VIN extraction (downloads PP-OCRv5 weights on first run)."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import get_settings
from app.services.ocr_engine import get_ocr_engine, reset_ocr_engine
from app.services.vin_extractor import VinExtractor


def main() -> None:
    reset_ocr_engine()
    settings = get_settings()
    print("backend=", settings.ocr_backend)
    print("det=", settings.paddle_det_model)
    print("rec=", settings.paddle_rec_model)
    print("Loading OCR engine (downloads models on first run)...")
    engine = get_ocr_engine(settings)
    print("OCR engine loaded:", type(engine).__name__)

    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        img,
        "1HGCM82633A004352",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    lines = engine.read(img)
    print("OCR lines:", [(line.text, round(line.confidence, 3)) for line in lines])

    result = VinExtractor(settings=settings, ocr=engine).extract_from_image(img)
    print(
        "extract:",
        result.status,
        result.vin,
        result.confidence_percent,
        result.message,
    )


if __name__ == "__main__":
    main()
