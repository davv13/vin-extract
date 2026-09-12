"""OCR backends for reading text from vehicle images.

Recommended default: PaddleOCR PP-OCRv5 (English-tuned recognition + server detection).
EasyOCR remains available as a lighter fallback.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    confidence: float


class OcrEngine(ABC):
    @abstractmethod
    def read(self, image: np.ndarray) -> list[OcrLine]:
        raise NotImplementedError


class EasyOcrEngine(OcrEngine):
    def __init__(self, langs: list[str], gpu: bool = False) -> None:
        import easyocr

        logger.info("Loading EasyOCR (langs=%s, gpu=%s)…", langs, gpu)
        self._reader = easyocr.Reader(langs, gpu=gpu)
        logger.info("EasyOCR ready.")

    def read(self, image: np.ndarray) -> list[OcrLine]:
        results = self._reader.readtext(image, detail=1, paragraph=False)
        lines: list[OcrLine] = []
        for item in results:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            conf = float(item[2])
            if text:
                lines.append(OcrLine(text=text, confidence=max(0.0, min(1.0, conf))))
        return lines


class PaddleOcrEngine(OcrEngine):
    """
    PaddleOCR PP-OCRv5 — best practical local OCR for VIN phone photos.

    Uses:
      - PP-OCRv5_server_det  → strongest text detection
      - en_PP-OCRv5_mobile_rec → English-optimized recognition (VIN charset)
      - text-line orientation → handles tilted phone shots
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        ocr_version: str = "PP-OCRv5",
        det_model: str = "PP-OCRv5_server_det",
        rec_model: str = "en_PP-OCRv5_mobile_rec",
        device: str = "cpu",
        use_doc_orientation: bool = True,
        use_textline_orientation: bool = True,
        enable_mkldnn: bool = False,
    ) -> None:
        # Avoid PaddlePaddle 3.3.x CPU crash: PIR + oneDNN incompatibility.
        # Must be set before importing / constructing PaddleOCR predictors.
        import os

        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        from paddleocr import PaddleOCR

        logger.info(
            "Loading PaddleOCR %s (lang=%s, det=%s, rec=%s, device=%s)…",
            ocr_version,
            lang,
            det_model,
            rec_model,
            device,
        )

        # PaddleOCR 3.x API (PP-OCRv5). Fall back to 2.x kwargs if needed.
        # enable_mkldnn=False is required on many CPU Windows installs.
        try:
            self._ocr = PaddleOCR(
                lang=lang,
                ocr_version=ocr_version,
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                use_doc_orientation_classify=use_doc_orientation,
                use_doc_unwarping=False,
                use_textline_orientation=use_textline_orientation,
                device=device,
                enable_mkldnn=enable_mkldnn,
            )
            self._api = "v3"
        except TypeError:
            logger.warning(
                "PaddleOCR 3.x kwargs rejected; falling back to classic API "
                "(install paddleocr>=3.0 for PP-OCRv5)."
            )
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                show_log=False,
                enable_mkldnn=enable_mkldnn,
            )
            self._api = "v2"

        logger.info("PaddleOCR ready (api=%s).", self._api)

    def read(self, image: np.ndarray) -> list[OcrLine]:
        if self._api == "v3":
            return self._read_v3(image)
        return self._read_v2(image)

    def _read_v3(self, image: np.ndarray) -> list[OcrLine]:
        results = self._ocr.predict(image)
        lines: list[OcrLine] = []
        if not results:
            return lines

        for res in results:
            data = self._result_to_dict(res)
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            for text, score in zip(texts, scores):
                text = str(text).strip()
                if not text:
                    continue
                conf = float(score)
                lines.append(OcrLine(text=text, confidence=max(0.0, min(1.0, conf))))
        return lines

    def _read_v2(self, image: np.ndarray) -> list[OcrLine]:
        result = self._ocr.ocr(image, cls=True)
        lines: list[OcrLine] = []
        if not result:
            return lines
        page = result[0] if result else None
        if not page:
            return lines
        for row in page:
            if not row or len(row) < 2:
                continue
            text_info = row[1]
            text = str(text_info[0]).strip()
            conf = float(text_info[1])
            if text:
                lines.append(OcrLine(text=text, confidence=max(0.0, min(1.0, conf))))
        return lines

    @staticmethod
    def _result_to_dict(res) -> dict:
        if isinstance(res, dict):
            return res
        if hasattr(res, "json") and isinstance(res.json, dict):
            # Some versions nest under "res"
            payload = res.json
            if "res" in payload and isinstance(payload["res"], dict):
                return payload["res"]
            return payload
        # Attribute-style Result objects
        out: dict = {}
        for key in ("rec_texts", "rec_scores"):
            if hasattr(res, key):
                out[key] = getattr(res, key)
        return out


_engine_lock = threading.Lock()
_engine: OcrEngine | None = None


def get_ocr_engine(settings: Settings | None = None) -> OcrEngine:
    """Lazy singleton OCR engine (models load once per process)."""
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine
        cfg = settings or get_settings()
        backend = cfg.ocr_backend.lower().strip()

        if backend in ("paddle", "ppocr", "pp-ocrv5", "paddleocr"):
            _engine = PaddleOcrEngine(
                lang=cfg.paddle_lang,
                ocr_version=cfg.paddle_ocr_version,
                det_model=cfg.paddle_det_model,
                rec_model=cfg.paddle_rec_model,
                device=cfg.paddle_device,
                use_doc_orientation=cfg.paddle_use_doc_orientation,
                use_textline_orientation=cfg.paddle_use_textline_orientation,
                enable_mkldnn=cfg.paddle_enable_mkldnn,
            )
        elif backend == "easyocr":
            langs = [p.strip() for p in cfg.easyocr_langs.split(",") if p.strip()] or ["en"]
            _engine = EasyOcrEngine(langs=langs, gpu=cfg.easyocr_gpu)
        else:
            raise ValueError(
                f"Unknown OCR_BACKEND={backend!r}. Use 'paddle' (recommended) or 'easyocr'."
            )
        return _engine


def reset_ocr_engine() -> None:
    """Test helper to clear the singleton."""
    global _engine
    with _engine_lock:
        _engine = None
