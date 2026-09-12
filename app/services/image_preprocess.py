"""Image preprocessing to improve OCR on phone photos of VIN plates/stickers."""

from __future__ import annotations

import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use JPEG, PNG, WEBP, BMP, or TIFF.")
    return image


def _resize_max(image: np.ndarray, max_side: int = 960) -> np.ndarray:
    """Downscale large phone photos for faster single-pass OCR (target <10s)."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def prepare_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Fast single-image prep: resize + light contrast boost.

    No multi-variant retries — one image in, one OCR pass.
    """
    resized = _resize_max(image)
    # Mild CLAHE on luminance keeps color for the detector while helping contrast
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    luminance = clahe.apply(luminance)
    return cv2.cvtColor(cv2.merge([luminance, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_for_ocr(image: np.ndarray) -> list[np.ndarray]:
    """Backward-compatible helper — returns a single prepared image."""
    return [prepare_image_for_ocr(image)]
