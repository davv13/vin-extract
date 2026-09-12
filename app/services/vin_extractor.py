"""End-to-end VIN extraction from an image."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from app.config import Settings, get_settings
from app.services.vin_validator import VinCandidate, extract_vin_candidates_from_text, looks_like_vin_label_context

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, str], None]

WAIT_10_MSG = "Taking longer than 10 seconds — please wait…"
WAIT_30_MSG = "Still working after 30 seconds — please keep waiting…"


@dataclass
class ExtractionResult:
    success: bool
    vin: str | None
    confidence: float
    confidence_percent: float
    status: str
    message: str
    check_digit_valid: bool | None
    candidates: list[dict]
    elapsed_seconds: float = 0.0


def _apply_context_boost(candidates: list[VinCandidate], texts: list[str]) -> list[VinCandidate]:
    """Slightly boost scores when surrounding OCR mentions VIN / chassis labels."""
    if not any(looks_like_vin_label_context(t) for t in texts):
        return candidates
    boosted: list[VinCandidate] = []
    for c in candidates:
        new_conf = min(1.0, round(c.final_confidence + 0.03, 4))
        boosted.append(
            VinCandidate(
                vin=c.vin,
                source_text=c.source_text,
                ocr_confidence=c.ocr_confidence,
                format_score=c.format_score,
                check_digit_valid=c.check_digit_valid,
                final_confidence=new_conf,
            )
        )
    return sorted(boosted, key=lambda x: x.final_confidence, reverse=True)


def _build_result(
    candidates: list[VinCandidate],
    threshold: float,
    elapsed_seconds: float = 0.0,
) -> ExtractionResult:
    if not candidates:
        return ExtractionResult(
            success=False,
            vin=None,
            confidence=0.0,
            confidence_percent=0.0,
            status="not_found",
            message=(
                "No VIN (16–17 character code) was detected. "
                "Please retake the photo closer to the VIN plate, with better lighting and focus."
            ),
            check_digit_valid=None,
            candidates=[],
            elapsed_seconds=round(elapsed_seconds, 2),
        )

    top = candidates[0]
    top_dicts = [
        {
            "vin": c.vin,
            "confidence": c.final_confidence,
            "confidence_percent": round(c.final_confidence * 100, 2),
            "ocr_confidence": c.ocr_confidence,
            "check_digit_valid": c.check_digit_valid,
        }
        for c in candidates[:5]
    ]

    conf = top.final_confidence
    percent = round(conf * 100, 2)

    if conf >= threshold:
        return ExtractionResult(
            success=True,
            vin=top.vin,
            confidence=conf,
            confidence_percent=percent,
            status="accepted",
            message="VIN extracted successfully.",
            check_digit_valid=top.check_digit_valid,
            candidates=top_dicts,
            elapsed_seconds=round(elapsed_seconds, 2),
        )

    # Below threshold → return once. Do NOT re-run OCR / extra passes.
    return ExtractionResult(
        success=False,
        vin=top.vin,
        confidence=conf,
        confidence_percent=percent,
        status="low_confidence",
        message=(
            f"A possible VIN was found but confidence is {percent:.1f}% "
            f"(required >= {threshold * 100:.0f}%). "
            "Please upload a clearer, higher-resolution photo of the VIN."
        ),
        check_digit_valid=top.check_digit_valid,
        candidates=top_dicts,
        elapsed_seconds=round(elapsed_seconds, 2),
    )


class _WaitNotifier:
    """Emit / log please-wait messages at 10s and 30s while OCR runs."""

    def __init__(self, on_progress: ProgressCallback | None = None) -> None:
        self._on_progress = on_progress
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vin-wait-notifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _emit(self, event: str, seconds: int, message: str) -> None:
        logger.info(message)
        if self._on_progress is not None:
            try:
                self._on_progress(event, seconds, message)
            except Exception:
                logger.exception("Progress callback failed")

    def _run(self) -> None:
        if self._stop.wait(10.0):
            return
        self._emit("waiting", 10, WAIT_10_MSG)
        if self._stop.wait(20.0):  # 30s total from start
            return
        self._emit("waiting", 30, WAIT_30_MSG)


class VinExtractor:
    def __init__(
        self,
        settings: Settings | None = None,
        ocr=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ocr = ocr

    def _engine(self):
        if self.ocr is not None:
            return self.ocr
        from app.services.ocr_engine import get_ocr_engine

        return get_ocr_engine(self.settings)

    def extract_from_bytes(
        self,
        image_bytes: bytes,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        from app.services.image_preprocess import decode_image_bytes

        image = decode_image_bytes(image_bytes)
        return self.extract_from_image(image, on_progress=on_progress)

    def extract_from_image(
        self,
        image,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """
        Single-pass OCR only.

        If confidence is below the threshold, return immediately — no extra
        preprocess / OCR retries.
        """
        from app.services.image_preprocess import prepare_image_for_ocr

        started = time.perf_counter()
        notifier = _WaitNotifier(on_progress=on_progress)
        notifier.start()
        try:
            if on_progress is not None:
                on_progress("started", 0, "Processing image…")

            prepared = prepare_image_for_ocr(image)
            engine = self._engine()
            threshold = self.settings.confidence_threshold

            try:
                lines = engine.read(prepared)
            except Exception:
                logger.exception("OCR failed")
                lines = []

            texts = [line.text for line in lines]
            confs = [line.confidence for line in lines]
            logger.info("OCR produced %s text fragments (single pass)", len(texts))

            candidates = extract_vin_candidates_from_text(texts, confs)
            candidates = _apply_context_boost(candidates, texts)
            elapsed = time.perf_counter() - started
            result = _build_result(candidates, threshold, elapsed_seconds=elapsed)

            if on_progress is not None:
                on_progress("done", int(elapsed), result.message)
            return result
        finally:
            notifier.stop()

    def extract_to_dict(self, image_bytes: bytes) -> dict:
        result = self.extract_from_bytes(image_bytes)
        return asdict(result)
