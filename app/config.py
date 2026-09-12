"""Application settings for the VIN extraction API."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VIN Extract API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Confidence threshold (0–1). At or above → accept VIN. No OCR re-runs below this.
    confidence_threshold: float = 0.80

    # OCR backend: "paddle" (PP-OCRv5, recommended) | "easyocr"
    ocr_backend: str = "paddle"

    # --- PaddleOCR / PP-OCRv5 (speed-oriented defaults for <10s target) ---
    paddle_lang: str = "en"
    paddle_ocr_version: str = "PP-OCRv5"
    paddle_det_model: str = "PP-OCRv5_mobile_det"
    paddle_rec_model: str = "en_PP-OCRv5_mobile_rec"
    # "cpu" or "gpu:0"
    paddle_device: str = "cpu"
    # Doc orientation is slow on CPU — off by default for latency
    paddle_use_doc_orientation: bool = False
    paddle_use_textline_orientation: bool = True
    # Keep False on Windows CPU — Paddle 3.3.x oneDNN/PIR crash workaround
    paddle_enable_mkldnn: bool = False

    # --- EasyOCR fallback ---
    easyocr_langs: str = "en"
    easyocr_gpu: bool = False

    # Max upload size in megabytes
    max_upload_mb: int = 15

    # Allowed image MIME types
    allowed_content_types: str = "image/jpeg,image/png,image/webp,image/bmp,image/tiff"

    # CORS — comma-separated origins, or "*" for all
    cors_origins: str = "*"

    @property
    def allowed_content_type_set(self) -> set[str]:
        return {t.strip().lower() for t in self.allowed_content_types.split(",") if t.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
