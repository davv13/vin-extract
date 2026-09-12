"""HTTP routes for VIN extraction."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.schemas import ExtractResponse, HealthResponse
from app.services.vin_extractor import VinExtractor

logger = logging.getLogger(__name__)

router = APIRouter()


def get_extractor(settings: Settings = Depends(get_settings)) -> VinExtractor:
    return VinExtractor(settings=settings)


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in settings.allowed_content_type_set:
        name = (file.filename or "").lower()
        if not name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported content type '{content_type}'. "
                    f"Allowed: {', '.join(sorted(settings.allowed_content_type_set))}"
                ),
            )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {settings.max_upload_mb} MB.",
        )
    return data


def _to_response(result) -> ExtractResponse:
    return ExtractResponse(
        success=result.success,
        vin=result.vin,
        confidence=result.confidence,
        confidence_percent=result.confidence_percent,
        status=result.status,
        message=result.message,
        check_digit_valid=result.check_digit_valid,
        candidates=result.candidates,
        elapsed_seconds=result.elapsed_seconds,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        ocr_backend=settings.ocr_backend,
        confidence_threshold=settings.confidence_threshold,
    )


@router.post(
    "/extract",
    response_model=ExtractResponse,
    tags=["vin"],
    summary="Extract VIN from an uploaded image (single OCR pass)",
    responses={
        400: {"description": "Invalid or unsupported image"},
        413: {"description": "Image too large"},
    },
)
async def extract_vin(
    file: UploadFile = File(..., description="Image containing a vehicle VIN"),
    settings: Settings = Depends(get_settings),
    extractor: VinExtractor = Depends(get_extractor),
) -> ExtractResponse:
    """
    Upload once → single OCR pass → JSON result.

    Below-threshold results are returned immediately (no OCR re-run).
    Server logs "please wait" at 10s and 30s if still processing.
    For live UI wait messages, prefer `POST /extract/stream`.
    """
    data = await _read_upload(file, settings)

    try:
        result = await run_in_threadpool(extractor.extract_from_bytes, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("VIN extraction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="VIN extraction failed due to an internal error.",
        ) from None

    return _to_response(result)


@router.post(
    "/extract/stream",
    tags=["vin"],
    summary="Extract VIN with live please-wait progress (SSE)",
)
async def extract_vin_stream(
    file: UploadFile = File(..., description="Image containing a vehicle VIN"),
    settings: Settings = Depends(get_settings),
    extractor: VinExtractor = Depends(get_extractor),
) -> StreamingResponse:
    """
    Server-Sent Events stream:

    - `{"type":"started","message":"..."}`
    - `{"type":"waiting","seconds":10,"message":"Taking longer than 10 seconds — please wait…"}`
    - `{"type":"waiting","seconds":30,"message":"Still working after 30 seconds — please keep waiting…"}`
    - `{"type":"result", ...ExtractResponse fields...}`
    """
    data = await _read_upload(file, settings)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_progress(event: str, seconds: int, message: str) -> None:
        payload = {"type": event, "seconds": seconds, "message": message}
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    async def event_generator() -> AsyncIterator[str]:
        task = asyncio.create_task(
            run_in_threadpool(
                extractor.extract_from_bytes,
                data,
                on_progress,
            )
        )
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.25)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except TimeoutError:
                    if task.done():
                        # drain any late progress events
                        while not queue.empty():
                            item = queue.get_nowait()
                            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                        break

            result = await task
            payload = _to_response(result).model_dump()
            payload["type"] = "result"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        except Exception:
            logger.exception("Streaming VIN extraction failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "detail": "VIN extraction failed due to an internal error.",
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
