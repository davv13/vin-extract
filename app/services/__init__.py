"""Service package."""

__all__ = ["ExtractionResult", "VinExtractor"]


def __getattr__(name: str):
    if name in ("ExtractionResult", "VinExtractor"):
        from app.services.vin_extractor import ExtractionResult, VinExtractor

        return {"ExtractionResult": ExtractionResult, "VinExtractor": VinExtractor}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
