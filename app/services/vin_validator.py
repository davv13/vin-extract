"""VIN format validation, normalization, and ISO 3779 check-digit verification."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ISO 3779 VIN alphabet — I, O, Q deliberately excluded
VIN_CHARSET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
VIN_CHAR_SET = frozenset(VIN_CHARSET)

# Transliteration values for check digit
_TRANSLITERATION: dict[str, int] = {
    **{str(d): d for d in range(10)},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}

_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)

# Raw VIN-like token: 16–17 chars from extended Latin + digits (before cleanup)
_RAW_VIN_RE = re.compile(r"[A-Za-z0-9IOQioq|]{16,17}")


@dataclass(frozen=True)
class VinCandidate:
    vin: str
    source_text: str
    ocr_confidence: float
    format_score: float
    check_digit_valid: bool | None
    final_confidence: float


def normalize_vin_chars(text: str) -> tuple[str, int]:
    """
    Uppercase and map illegal OCR lookalikes to legal VIN characters.

    Returns (normalized, substitution_count) where substitutions are I/O/Q/| → digit.
    """
    out: list[str] = []
    subs = 0
    for ch in text.upper():
        if ch in VIN_CHAR_SET:
            out.append(ch)
        elif ch in ("I", "|", "!"):
            out.append("1")
            subs += 1
        elif ch in ("O", "Q"):
            out.append("0")
            subs += 1
        # drop spaces / punctuation / unknown symbols
    return "".join(out), subs


def normalize_token(token: str) -> str:
    """Normalize a single OCR token into a VIN-shaped string."""
    cleaned = re.sub(r"[^A-Za-z0-9IOQioq|]", "", token)
    vin, _ = normalize_vin_chars(cleaned)
    return vin


def digit_count(vin: str) -> int:
    return sum(ch.isdigit() for ch in vin)


def looks_like_label_garbage(vin: str, source: str) -> bool:
    """
    Reject OCR slices of English label phrases that only look like VINs
    after I/O→1/0 substitutions (e.g. DATE OF MANUFACT → NTHEDATE0FMANUFAC).
    """
    if digit_count(vin) < 4:
        return True
    # Too many letter→digit substitutions usually means English words, not a VIN
    _, subs = normalize_vin_chars(re.sub(r"[^A-Za-z0-9IOQioq|]", "", source))
    if subs >= 4:
        return True
    # Dense vowel runs are rare in real VINs and common in English fragments
    if re.search(r"[AEU]{3,}", vin):
        return True
    lowered = re.sub(r"[^A-Z0-9]", "", source.upper())
    ban_frags = (
        "DATE",
        "MANUFAC",
        "MOTOR",
        "CORP",
        "VEHICLE",
        "WEIGHT",
        "POUNDS",
        "PASSENGER",
        "CAPACITY",
        "GAWR",
        "GVWR",
        "REAR",
        "FRONT",
        "ASSEMBLED",
        "STANDARDS",
        "FEDERAL",
        "CONFORM",
        "SAFETY",
        "MADEIN",
        "GERMANY",
        "JAPAN",
        "LBS",
        "MDH",
    )
    if any(frag in lowered for frag in ban_frags):
        return True
    # Trailing English conjunctions / prepositions after digit blocks
    if re.search(r"(AND|THE|FOR|OF)$", vin):
        return True
    return False


_VIN_LABEL_RE = re.compile(
    r"(?:VIN|V\.?I\.?N\.?)\s*[:#-]?\s*([A-HJ-NPR-Z0-9IOQioq|]{16,20})",
    re.IGNORECASE,
)


def extract_labeled_vins(text: str) -> list[str]:
    """Return raw tokens explicitly labeled as VIN in OCR text."""
    return [m.group(1) for m in _VIN_LABEL_RE.finditer(text)]


def looks_like_vin_label_context(text: str) -> bool:
    """Boost awareness when OCR text mentions VIN / chassis / WMI labels."""
    lowered = text.lower()
    keywords = ("vin", "vehicle identification", "chassis", "fahrgestell", "номер кузова")
    return any(k in lowered for k in keywords)


def is_valid_charset(vin: str) -> bool:
    return bool(vin) and all(c in VIN_CHAR_SET for c in vin) and len(vin) in (16, 17)


def compute_check_digit(vin17: str) -> str:
    """Return expected check digit character for a 17-char VIN (positions 1–17)."""
    if len(vin17) != 17:
        raise ValueError("Check digit requires a 17-character VIN")
    total = 0
    for i, ch in enumerate(vin17):
        if i == 8:
            continue
        value = _TRANSLITERATION.get(ch)
        if value is None:
            raise ValueError(f"Invalid VIN character: {ch}")
        total += value * _WEIGHTS[i]
    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def verify_check_digit(vin: str) -> bool | None:
    """Validate ISO 3779 check digit for 17-char VINs. Returns None for 16-char."""
    if len(vin) == 16:
        return None
    if len(vin) != 17 or not is_valid_charset(vin):
        return False
    try:
        return vin[8] == compute_check_digit(vin)
    except ValueError:
        return False


def format_score(vin: str) -> float:
    """Heuristic quality score based on charset, length, digits, and check digit."""
    if not is_valid_charset(vin):
        return 0.0
    digits = digit_count(vin)
    if digits < 4:
        return 0.15

    score = 0.45
    if len(vin) == 17:
        score += 0.15
        cd = verify_check_digit(vin)
        if cd is True:
            score += 0.20
        elif cd is False:
            score -= 0.15
    elif len(vin) == 16:
        score += 0.08

    # Real VINs are digit-heavy in the serial section
    if digits >= 6:
        score += 0.12
    elif digits >= 4:
        score += 0.05

    return max(0.0, min(1.0, score))


def combine_confidence(ocr_confidence: float, vin: str, substitutions: int = 0) -> float:
    """
    Blend OCR confidence with structural VIN validity.

    Weights favor OCR when structure is solid; check-digit failure / OCR
    letter→digit substitutions pull the score down.
    """
    ocr = max(0.0, min(1.0, ocr_confidence))
    fmt = format_score(vin)
    blended = 0.65 * ocr + 0.35 * fmt

    if substitutions:
        blended -= min(0.35, 0.06 * substitutions)

    digits = digit_count(vin)
    solid = (
        is_valid_charset(vin)
        and len(vin) == 17
        and verify_check_digit(vin) is True
        and digits >= 5
        and substitutions <= 2
    )
    if solid:
        blended = max(blended, min(0.95, ocr * 0.85 + 0.12))
    elif is_valid_charset(vin) and verify_check_digit(vin) is False:
        blended = min(blended, 0.85)
    elif digits < 5 or substitutions >= 3:
        blended = min(blended, 0.82)

    return round(max(0.0, min(1.0, blended)), 4)


def extract_vin_candidates_from_text(
    texts: list[str],
    confidences: list[float],
) -> list[VinCandidate]:
    """
    Find VIN-like strings in OCR line results.

    Also scans concatenated nearby lines (VIN sometimes split across boxes).
    """
    if len(texts) != len(confidences):
        raise ValueError("texts and confidences must have the same length")

    candidates: dict[str, VinCandidate] = {}

    def consider(raw: str, conf: float, *, from_label: bool = False) -> None:
        # Prefer explicit "VIN: XXXXX" tokens
        for labeled in extract_labeled_vins(raw):
            _add_normalized(labeled, conf, candidates, from_label=True)
        # Direct normalize of contiguous alphanumeric runs
        for match in _RAW_VIN_RE.finditer(re.sub(r"\s+", "", raw)):
            _add_normalized(match.group(0), conf, candidates, from_label=from_label)
        # Also try whole-string normalize (handles embedded noise)
        compact = re.sub(r"[^A-Za-z0-9IOQioq|]", "", raw)
        if len(compact) >= 16:
            for i in range(0, len(compact) - 15):
                for length in (17, 16):
                    if i + length <= len(compact):
                        _add_normalized(
                            compact[i : i + length],
                            conf,
                            candidates,
                            from_label=from_label,
                        )

    for text, conf in zip(texts, confidences):
        consider(text, conf)

    # Sliding window over joined OCR lines (helps when VIN is fragmented)
    if texts:
        joined = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences)
        consider(joined, avg_conf)
        # Join without spaces
        consider("".join(texts), avg_conf)

    ranked = sorted(
        candidates.values(),
        key=lambda c: (c.final_confidence, c.check_digit_valid is True),
        reverse=True,
    )
    return ranked


def _add_normalized(
    token: str,
    ocr_conf: float,
    store: dict[str, VinCandidate],
    *,
    from_label: bool = False,
) -> None:
    cleaned = re.sub(r"[^A-Za-z0-9IOQioq|]", "", token)
    vin, subs = normalize_vin_chars(cleaned)
    if len(vin) not in (16, 17) or not is_valid_charset(vin):
        return
    if looks_like_label_garbage(vin, token):
        return
    final = combine_confidence(ocr_conf, vin, substitutions=subs)
    if from_label:
        final = min(1.0, round(final + 0.08, 4))
    existing = store.get(vin)
    if existing is None or final > existing.final_confidence:
        store[vin] = VinCandidate(
            vin=vin,
            source_text=token,
            ocr_confidence=round(ocr_conf, 4),
            format_score=round(format_score(vin), 4),
            check_digit_valid=verify_check_digit(vin),
            final_confidence=final,
        )
