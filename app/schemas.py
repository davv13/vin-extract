from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    ocr_backend: str
    confidence_threshold: float


class ExtractResponse(BaseModel):
    """Result of VIN extraction from an uploaded image."""

    success: bool = Field(
        ...,
        description="True when a VIN was found with confidence at or above the threshold.",
    )
    vin: str | None = Field(
        None,
        description="Extracted VIN (16 or 17 alphanumeric characters), uppercase.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the returned VIN (0–1).",
    )
    confidence_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Same confidence as a percentage (0–100).",
    )
    status: str = Field(
        ...,
        description=(
            "accepted — VIN returned with confidence >= threshold; "
            "low_confidence — candidate found but below threshold (no OCR re-run); "
            "not_found — no plausible VIN detected."
        ),
    )
    message: str = Field(..., description="Human-readable guidance for the client/UI.")
    check_digit_valid: bool | None = Field(
        None,
        description="For 17-char VINs: whether ISO 3779 check digit is valid. Null for 16-char.",
    )
    candidates: list[dict] = Field(
        default_factory=list,
        description="Alternative VIN candidates with scores (for debugging / UI hints).",
    )
    elapsed_seconds: float = Field(
        0.0,
        description="Server-side processing time for this request in seconds.",
    )


class ErrorResponse(BaseModel):
    detail: str
