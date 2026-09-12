"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.services.ocr_engine import get_ocr_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting %s v%s (OCR=%s, threshold=%.0f%%)",
        settings.app_name,
        settings.app_version,
        settings.ocr_backend,
        settings.confidence_threshold * 100,
    )
    # Warm OCR models at startup so the first request is not slow
    try:
        get_ocr_engine(settings)
    except Exception:
        logger.exception(
            "OCR engine failed to load at startup. "
            "The API will still start; /extract will error until models are available."
        )
    yield
    logger.info("Shutting down VIN Extract API")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Upload a phone photo that contains a vehicle VIN (16–17 character "
            "alphanumeric code). Returns the VIN with a confidence score. "
            "If confidence is below the configured threshold (default 90%), "
            "the API asks for a clearer image."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
