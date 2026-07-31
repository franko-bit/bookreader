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

AI model integration is intentionally isolated in `backend/app/services.py`; replace the lightweight MVP implementations with hosted or local models as infrastructure becomes available.
