"""
Example client for the VIN Extract API.

Usage:
  python examples/client_example.py path/to/photo.jpg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/extract"


def extract_vin(image_path: str | Path, api_url: str = API_URL) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime = "image/jpeg"
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"

    with path.open("rb") as f:
        response = requests.post(
            api_url,
            files={"file": (path.name, f, mime)},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/client_example.py <image_path>")
        sys.exit(1)

    result = extract_vin(sys.argv[1])
    print(json.dumps(result, indent=2))

    if result.get("success"):
        print(f"\nVIN: {result['vin']} ({result['confidence_percent']}% confidence)")
    elif result.get("status") == "low_confidence":
        print(f"\nLow confidence — ask user to re-upload. Candidate: {result.get('vin')}")
    else:
        print("\nNo VIN found — ask user for a clearer photo.")


if __name__ == "__main__":
    main()
