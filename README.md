# VIN Extract API

Local image → VIN extraction service for websites and mobile apps. Upload a phone photo that contains a vehicle identification number; the API returns the **16–17 character VIN** with a confidence score.

If confidence is **≥ 80%** (configurable), the VIN is accepted. Below that, the API returns immediately with the best candidate and asks for a clearer photo — **no OCR re-run**.

While processing, the server logs (and `/extract/stream` pushes) please-wait messages at **10s** and **30s**.

## Recommended local model

**Use PaddleOCR PP-OCRv5** (default in this project).

| Role | Model | Why |
|------|--------|-----|
| Text detection | `PP-OCRv5_server_det` | Highest-quality local detector for small/angled plate text |
| Text recognition | `en_PP-OCRv5_mobile_rec` | English-tuned recognizer (best match for VIN A–Z / 0–9) |
| Orientation | doc + text-line orientation | Handles rotated / upside-down phone photos |

Why not EasyOCR or a big vision-LLM?

- PP-OCRv5 beats EasyOCR by a large margin on OCR benchmarks (~5× lower edit distance).
- Large VLMs (Qwen2.5-VL, etc.) can help on ruined photos but are slower, need a GPU, and may **hallucinate** characters — bad for a 17-char exact VIN.
- For VIN, specialized OCR + check-digit validation is the most reliable local stack.

Set `OCR_BACKEND=easyocr` only if you cannot install PaddlePaddle.

## Features

- FastAPI REST endpoint (`POST /extract`)
- **PP-OCRv5** local OCR by default (EasyOCR optional fallback)
- Image preprocessing for phone photos (contrast, denoise, thresholding)
- VIN normalization (maps OCR lookalikes `I→1`, `O/Q→0`)
- ISO 3779 check-digit validation for 17-character VINs
- Confidence blending: OCR score + format / check-digit quality
- CORS enabled for frontend integration

## Project layout

```
vin-extract/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings (.env)
│   ├── schemas.py              # Response models
│   ├── api/routes.py           # /health, /extract
│   └── services/
│       ├── image_preprocess.py
│       ├── ocr_engine.py
│       ├── vin_validator.py
│       └── vin_extractor.py
├── examples/client_example.py
├── tests/
├── requirements.txt
├── run.py
└── .env.example
```

## Setup

Python **3.10–3.12** recommended (PaddlePaddle may not support 3.13 yet).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 1) Install PaddlePaddle for your OS/CPU-or-GPU:
#    https://www.paddlepaddle.org.cn/install/quick
# CPU example:
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2) Install the API + PaddleOCR
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

First run downloads PP-OCRv5 weights automatically.

## Run the API

```bash
python run.py
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs  
- Health: `GET http://127.0.0.1:8000/health`

## API

### `POST /extract`

Multipart form field: `file` (JPEG / PNG / WEBP / BMP / TIFF).

**Success (confidence ≥ threshold)**

```json
{
  "success": true,
  "vin": "1HGCM82633A004352",
  "confidence": 0.9345,
  "confidence_percent": 93.45,
  "status": "accepted",
  "message": "VIN extracted successfully.",
  "check_digit_valid": true,
  "candidates": [...]
}
```

**Low confidence — ask user to re-upload**

```json
{
  "success": false,
  "vin": "1HGCM82633A004352",
  "confidence": 0.72,
  "confidence_percent": 72.0,
  "status": "low_confidence",
  "message": "A possible VIN was found but confidence is 72.0% (required ≥ 90%). Please upload a clearer, higher-resolution photo of the VIN.",
  "check_digit_valid": true,
  "candidates": [...]
}
```

**Not found**

```json
{
  "success": false,
  "vin": null,
  "confidence": 0.0,
  "confidence_percent": 0.0,
  "status": "not_found",
  "message": "No VIN (16–17 character code) was detected. ...",
  "check_digit_valid": null,
  "candidates": []
}
```

### Frontend integration (JavaScript)

```javascript
async function extractVin(file) {
  const form = new FormData();
  form.append("file", file); // from <input type="file"> or camera capture

  const res = await fetch("http://YOUR_SERVER:8000/extract", {
    method: "POST",
    body: form,
  });
  const data = await res.json();

  if (data.success) {
    return { ok: true, vin: data.vin, confidence: data.confidence_percent };
  }
  if (data.status === "low_confidence" || data.status === "not_found") {
    return { ok: false, message: data.message, vin: data.vin ?? null };
  }
  throw new Error(data.detail || "Extraction failed");
}
```

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/extract" -F "file=@photo.jpg"
```

### Python client

```bash
pip install requests
python examples/client_example.py photo.jpg
```

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.80` | Accept when confidence ≥ this (no OCR re-run below) |
| `OCR_BACKEND` | `paddle` | `paddle` (PP-OCRv5) or `easyocr` |
| `PADDLE_DET_MODEL` | `PP-OCRv5_mobile_det` | Detection model (mobile = faster) |
| `PADDLE_REC_MODEL` | `en_PP-OCRv5_mobile_rec` | English recognition model |
| `PADDLE_DEVICE` | `cpu` | `cpu` or `gpu:0` |
| `MAX_UPLOAD_MB` | `15` | Max upload size |
| `CORS_ORIGINS` | `*` | Allowed browser origins |

Higher accuracy / slower: set `PADDLE_DET_MODEL=PP-OCRv5_server_det`.

## VIN rules used

- Length: **16 or 17** characters
- Alphabet: `A–H`, `J–N`, `P`, `R–Z`, `0–9` (no `I`, `O`, `Q`)
- 17-character VINs: ISO 3779 check digit at position 9

## Tests

```bash
pip install pytest
pytest -q
```

## Notes for production

- Put the API behind HTTPS and restrict `CORS_ORIGINS` to your site.
- Models load once at startup; keep the process warm (or use a worker pool).
- Prefer `PADDLE_DEVICE=gpu:0` when a CUDA GPU is available.
- As local models improve, swap backends in `ocr_engine.py` without changing the `/extract` contract.
