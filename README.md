# Voice Reader

A no-account, browser-based PDF reader and speech-coaching MVP. Upload a PDF, select text for study tools, then record yourself reading it for immediate pace and fluency feedback.

## Run locally

### Frontend

```powershell
npm install
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`).

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The frontend uses useful local fallbacks when the backend is not running. Set `VITE_API_URL` to point at a different backend URL if needed.

## Current MVP scope

- PDF upload, reading position, bookmarks, selected-text workspace, and local persistence.
- Translation/summarization/read-aloud controls (browser-native fallbacks, backend-ready APIs).
- Browser recording, replay/deletion, voice notes, WPM and transcript-comparison feedback.
- FastAPI endpoints with replaceable AI service stubs.

AI model integration is isolated in `backend/app/services.py`. Translation uses Hugging Face's `facebook/nllb-200-distilled-600M` model for English, French, and Kinyarwanda. On the first translation request, the backend downloads and initializes the model; later requests reuse its local Hugging Face cache. This requires network access and substantial RAM (a GPU is recommended for responsive inference).

### Backend translation configuration

Set these environment variables in `backend` to control local/remote translation behavior:

- `NLLB_FAST_MODE=1` (default) enables faster local generation with smaller beams.
- `NLLB_WARMUP=1` (default) runs a quick model warm-up on startup.
- `HF_INFERENCE_API=1` forces translation through the Hugging Face Inference API instead of the local model.
- `HF_API_TOKEN=<your_token>` is required when using the remote inference API.
- `HF_INFERENCE_URL=<url>` optionally overrides the Hugging Face inference endpoint.

When local model loading fails, the backend falls back to the remote Hugging Face Inference API if `HF_API_TOKEN` is configured.
