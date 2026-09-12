"""Tests for response building / threshold logic."""

from app.services.vin_extractor import WAIT_10_MSG, WAIT_30_MSG, _build_result
from app.services.vin_validator import VinCandidate


def _cand(vin: str, conf: float) -> VinCandidate:
    return VinCandidate(
        vin=vin,
        source_text=vin,
        ocr_confidence=conf,
        format_score=0.9,
        check_digit_valid=True,
        final_confidence=conf,
    )


def test_accepted_at_80_percent():
    result = _build_result([_cand("1HGCM82633A004352", 0.80)], threshold=0.80)
    assert result.success is True
    assert result.status == "accepted"


def test_low_confidence_no_retry_message():
    result = _build_result([_cand("1HGCM82633A004352", 0.75)], threshold=0.80)
    assert result.success is False
    assert result.status == "low_confidence"
    assert result.vin == "1HGCM82633A004352"


def test_not_found():
    result = _build_result([], threshold=0.80)
    assert result.success is False
    assert result.status == "not_found"
    assert result.vin is None


def test_wait_messages_defined():
    assert "10 seconds" in WAIT_10_MSG
    assert "30 seconds" in WAIT_30_MSG
