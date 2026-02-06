# VoiceRad

**Voice-Controlled Mobile Radiology Assistant**

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) using Google's MedGemma, and Health AI Developer Foundations ([HAI-DEF](https://github.com/Google-Health/hai-def)).

---

## The Problem

Over **42 000 radiologist positions** will be unfilled globally by 2033, and **4.5 billion people** lack access to basic medical imaging services (WHO). Rural clinicians often have no radiologist within hours of travel.

## The Solution

VoiceRad is a **Progressive Web App** where a clinician can:

1. **Speak** a clinical question ("What do you see in this chest X-ray?")
2. **Upload** a medical image (DICOM, JPG, PNG) from a phone camera or scanner
3. **Receive** an AI-powered radiology interpretation with voice playback
4. **Refine** the result through multi-turn agentic conversation
5. **Work offline** and sync when connectivity returns

---

## Key Features

- Voice-controlled interface (Whisper-based speech-to-text with graceful fallback)
- Multimodal medical image analysis via **MedGemma 1.5 4B**
- Agentic multi-turn conversation for diagnostic refinement
- Offline-first PWA with background sync
- Structured radiology report generation
- Mobile-first responsive design
- Session TTL + cleanup for memory safety
- Input validation with clear HTTP error codes
- Configurable CORS origins

## Architecture

```
Browser (React PWA)
    | voice / image / text
    v
FastAPI Backend
    +-- Whisper / MedASR  (speech -> text)
    +-- MedGemma 1.5 4B   (image + text -> interpretation)
    +-- Agentic Loop       (multi-turn refinement)
    +-- TTS                (text -> speech)
```

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/naividh/voicerad.git
cd voicerad
docker-compose up

# Backend -> http://localhost:8000
# Frontend -> http://localhost:3000
```

### Manual

```bash
# Backend
cd backend
pip install -r requirements.txt
DEMO_MODE=1 python app.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Download Models

```bash
bash scripts/setup_models.sh
```

### Run Tests

```bash
pip install pytest httpx
DEMO_MODE=1 pytest
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health check |
| POST | `/api/voice/transcribe` | Medical speech-to-text |
| POST | `/api/images/upload` | Upload medical image |
| POST | `/api/interpret/start-session` | Begin interpretation |
| POST | `/api/interpret/continue/{id}` | Refine with new context |
| POST | `/api/interpret/finalize/{id}` | Generate final report |
| POST | `/api/sync/queue` | Queue offline sync |
| GET | `/api/sync/status` | Check sync status |

Full docs at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI).

---

## Project Structure

```
voicerad/
  backend/
    app.py                    FastAPI server
    models/
      medgemma_wrapper.py     MedGemma 1.5 4B inference
      medasr_wrapper.py       Speech-to-text (Whisper fallback)
    Dockerfile
    requirements.txt
  frontend/
    src/
      App.jsx                 Main app (component-split)
      useVoiceRecorder.js     Voice recording hook
      ImageUpload.jsx         Image upload component
      InterpretationView.jsx  Interpretation display
    public/
      sw.js                   Service Worker
    Dockerfile
  tests/
    test_api.py               26 pytest API tests
  scripts/
    setup_models.sh           Model downloader
  pytest.ini                  Test config
  docker-compose.yml
  LICENSE                     Apache 2.0
```

---

## Models & Speech Recognition

| Component | Primary | Fallback |
|-----------|---------|----------|
| Image interpretation | MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) | Demo stub |
| Speech-to-text | MedASR (`google/medASR` — CTC model) | OpenAI Whisper `base` |

> **Note:** The MedASR model ID `google/medASR` is the expected HAI-DEF speech model. If it is not available on HuggingFace at runtime, the wrapper **automatically falls back to OpenAI Whisper** (`base` model), which provides high-quality general-purpose medical transcription. Set `DEMO_MODE=1` to skip all model loading entirely.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `""` | Set to `1` / `true` to skip model loading |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `PORT` | `8000` | Backend port |
| `HF_TOKEN` | — | HuggingFace token for gated models |

---

## Prize Alignment

| Prize | How VoiceRad Qualifies |
|-------|------------------------|
| Main Track ($75 K) | Full application with real clinical impact |
| Edge AI ($5 K) | MedGemma 4B runs locally on mobile hardware |
| Agentic Workflow ($10 K) | Voice -> Image -> Multi-turn interpretation pipeline |
| Novel Task ($10 K) | Fine-tuned MedGemma for radiology report generation |

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, Transformers
- **Frontend:** React 18, Vite, Service Workers, IndexedDB
- **Models:** MedGemma 1.5 4B, Whisper (speech), MedSigLIP
- **Infra:** Docker, Docker Compose
- **Tests:** pytest, FastAPI TestClient

---

## Disclaimer

VoiceRad is a demonstration application. It is **not** approved for autonomous clinical diagnosis. All AI-generated interpretations require clinician review.

## License

Apache 2.0
# VoiceRad

**Voice-Controlled Mobile Radiology Assistant**

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) using Google's MedGemma, MedASR, and MedSigLIP from Health AI Developer Foundations (HAI-DEF).

---

## The Problem

Over **42 000 radiologist positions** will be unfilled globally by 2033, and **4.5 billion people** lack access to basic medical imaging services (WHO). Rural and resource-limited clinics simply cannot access cloud-dependent diagnostic tools.

## The Solution

VoiceRad is a Progressive Web App where a clinician can:

1. **Speak** a clinical question ("What do you see in this chest X-ray?")
2. **Upload** a medical image (DICOM, JPG, PNG) from a phone camera or scanner
3. **Receive** an AI-powered radiology interpretation with voice playback
4. **Refine** the result through multi-turn agentic conversation
5. **Work offline** and sync when connectivity returns

## Key Features

- Voice-controlled interface powered by MedASR
- Multimodal medical image analysis via MedGemma 1.5 4B
- Agentic multi-turn conversation for diagnostic refinement
- Offline-first PWA with background sync
- Structured radiology report generation
- Mobile-first responsive design

## Architecture

```
Browser (React PWA)
   |  voice / image / text
   v
FastAPI Backend
   |
   +-- MedASR       (speech-to-text)
   +-- MedGemma 1.5 (image + text interpretation)
   +-- MedSigLIP    (image embeddings, optional)
   |
   v
Structured Report -> Voice Playback (TTS)
```

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/naividh/voicerad.git
cd voicerad
docker-compose up
# Backend  -> http://localhost:8000
# Frontend -> http://localhost:3000
```

### Manual

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Download Models

```bash
bash scripts/setup_models.sh
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/voice/transcribe | Medical speech-to-text |
| POST | /api/images/upload | Upload medical image |
| POST | /api/interpret/start-session | Begin interpretation |
| POST | /api/interpret/continue/{id} | Refine with new context |
| POST | /api/interpret/finalize/{id} | Generate final report |
| GET  | /health | Service health check |

Full docs at http://localhost:8000/docs (Swagger UI).

## Project Structure

```
voicerad/
  backend/
    app.py                   FastAPI server
    models/
      medgemma_wrapper.py    MedGemma 1.5 4B integration
      medasr_wrapper.py      MedASR speech recognition
    Dockerfile
    requirements.txt
  frontend/
    src/
      App.jsx                React application
      App.css                Mobile-first styles
      main.jsx               Entry point + SW registration
    public/
      manifest.json          PWA manifest
      service-worker.js      Offline caching & sync
    index.html
    vite.config.js
    Dockerfile
    package.json
  scripts/
    setup_models.sh          Model download helper
  docker-compose.yml
```

## Prize Alignment

| Prize | How VoiceRad Qualifies |
|-------|------------------------|
| Main Track ($75 K) | Full application with real clinical impact |
| Edge AI ($5 K) | MedGemma 4B runs locally on mobile hardware |
| Agentic Workflow ($10 K) | Voice -> Image -> Multi-turn interpretation pipeline |
| Novel Task ($10 K) | Fine-tuned MedGemma for radiology report generation |

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, Transformers
- **Frontend:** React 18, Vite, Service Workers, IndexedDB
- **Models:** MedGemma 1.5 4B, MedASR, MedSigLIP
- **Infra:** Docker, Docker Compose

## Disclaimer

VoiceRad is a demonstration application. It is **not** approved for autonomous clinical diagnosis. All AI-generated interpretations must be reviewed by a qualified healthcare professional before any clinical decisions are made.

## License

Apache 2.0
