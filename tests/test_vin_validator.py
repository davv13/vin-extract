"""Unit tests for VIN validation / candidate extraction (no OCR download required)."""

from app.services.vin_validator import (
    combine_confidence,
    compute_check_digit,
    digit_count,
    extract_vin_candidates_from_text,
    normalize_token,
    verify_check_digit,
)


def test_normalize_maps_illegal_lookalikes():
    assert normalize_token("1HGCM82633A004352") == "1HGCM82633A004352"
    # I→1, O→0, Q→0
    assert normalize_token("ABCIOQ12345678901") == "ABC10012345678901"
    assert "I" not in normalize_token("ABCIOQ12345678901")
    assert "O" not in normalize_token("ABCIOQ12345678901")
    assert normalize_token("IHGCM82633AOO4352") == "1HGCM82633A004352"


def test_check_digit_known_vin():
    # Well-known valid sample VIN (Honda Accord example used in many tutorials)
    vin = "1HGCM82633A004352"
    assert len(vin) == 17
    assert compute_check_digit(vin) == vin[8]
    assert verify_check_digit(vin) is True


def test_check_digit_invalid():
    vin = "1HGCM82633A004352"
    bad = vin[:8] + "0" + vin[9:]  # wrong check digit (unless it coincides)
    if bad[8] != compute_check_digit(bad):
        assert verify_check_digit(bad) is False


def test_sixteen_char_has_no_check_digit():
    vin16 = "1HGCM82633A00435"
    assert verify_check_digit(vin16) is None


def test_extract_from_noisy_text():
    texts = [
        "Manufacturer label",
        "VIN: 1HGCM82633A004352",
        "Made in USA",
    ]
    confs = [0.9, 0.92, 0.8]
    cands = extract_vin_candidates_from_text(texts, confs)
    assert cands
    assert cands[0].vin == "1HGCM82633A004352"
    assert cands[0].final_confidence > 0.5


def test_extract_with_ocr_lookalikes():
    # O and I injected illegaly
    texts = ["VIN IHGCM82633AOO4352"]
    confs = [0.88]
    cands = extract_vin_candidates_from_text(texts, confs)
    assert cands
    assert cands[0].vin == "1HGCM82633A004352"


def test_rejects_label_phrase_false_positives():
    # "ON THE DATE OF MANUFACT..." must not become a VIN
    texts = ["ON THE DATE OF MANUFACTURE"]
    confs = [0.99]
    cands = extract_vin_candidates_from_text(texts, confs)
    vins = {c.vin for c in cands}
    assert "NTHEDATE0FMANUFAC" not in vins


def test_rejects_toyota_motor_corp_garbage():
    texts = ["TOYOTA MOTOR CORP"]
    confs = [0.99]
    cands = extract_vin_candidates_from_text(texts, confs)
    assert all(digit_count(c.vin) >= 4 for c in cands)


def test_rejects_gawr_rear_false_positive():
    texts = ["GAWR REAR 850 1874", "VIN: 1C3AN75NX5X042701"]
    confs = [0.99, 0.95]
    cands = extract_vin_candidates_from_text(texts, confs)
    vins = [c.vin for c in cands]
    assert "AWRREAR8501874AND" not in vins
    assert cands[0].vin == "1C3AN75NX5X042701"
