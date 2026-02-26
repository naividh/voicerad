# VoiceRad

**Voice-Controlled Mobile Radiology Assistant with Clinical Safety Rails**

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) using Google's [MedGemma](https://ai.google.dev/gemma/docs/medgemma) and [Health AI Developer Foundations (HAI-DEF)](https://developers.google.com/health-ai-developer-foundations).

> **Kaggle Resources:** [Competition Writeup](https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/voicerad-voice-controlled-mobile-radiology-assist) | [Kaggle Notebook](https://www.kaggle.com/code/naivedhyaagrawal/notebookd31da50165)

---

## The Problem

Over **42,000 radiologist positions** will be unfilled globally by 2033, and **4.5 billion people** lack access to basic medical imaging services (WHO). Rural clinicians often have no radiologist within hours of travel, leading to **up to 30% error rates** in chest X-ray interpretation.

## The Solution

VoiceRad is a **voice-first, offline-capable PWA** that lets rural clinicians upload a chest X-ray, ask questions by voice, and receive AI-assisted interpretations powered by **MedGemma 4B** with built-in clinical safety rails.

## Key Differentiators

- **Voice-first design** for hands-free clinical use (Whisper STT + edge-tts audio output)
- **Clinical safety rails** with deterministic triage (CRITICAL/URGENT/ROUTINE/NORMAL), model-calibrated confidence scoring, and referral triggers
- **Multi-turn agentic conversation** where clinicians refine interpretations with additional clinical context
- **Offline-first architecture** with IndexedDB queueing and background sync
- **DICOM native** support with metadata extraction and conversion

## Architecture

```
[Voice Input] --> [Whisper STT] --> [FastAPI Backend]
[Image Upload] --> [DICOM/PNG/JPG] ---|
                                       v
                              [MedGemma 4B] (4-bit NF4 quantized)
                                       |
                                       v
                              [Clinical Safety Engine]
                              (triage + confidence + referral)
                                       |
                                       v
                              [React PWA Frontend]
                              (SafetyBanner + TTS playback)
```

## Performance

| Metric | Value |
|--------|-------|
| Model | MedGemma 4B (google/medgemma-4b-it) |
| Quantization | 4-bit NF4 via BitsAndBytes |
| VRAM usage | ~3 GB (down from ~8 GB float16) |
| Inference (T4 GPU) | 7-10s per interpretation |
| Confidence scoring | Model log-probability based |
| Safety engine | Deterministic, regex + threshold based |
| Critical finding detection | ACR-aligned pattern matching |
| Triage levels | CRITICAL / URGENT / ROUTINE / NORMAL |

## Clinical Safety Rails

Every AI interpretation passes through a **deterministic safety engine** that cannot be bypassed by the model. Safety is monotonically increasing -- more information can only make the system MORE cautious, never less.

### Confidence Scoring (v1.3.0)

VoiceRad uses **model log-probability based confidence** when available, falling back to text heuristics. The `confidence_source` field tracks which method was used:

- **model_logprob**: Derived from mean token-level probability during generation. More reliable.
- **text_heuristic**: Estimated from response structure, length, and anatomical specificity. Used when log-probs unavailable.

### Triage Classification

| Level | Color | Action Required |
|-------|-------|-----------------|
| CRITICAL | Red (pulsing) | Immediate radiologist review |
| URGENT | Orange | Review within 1 hour |
| ROUTINE | Yellow | Standard review queue |
| NORMAL | Green | No acute findings |

### Hedging Language Policy (v1.3.0)

Appropriately cautious language ("cannot exclude", "clinical correlation recommended") is treated as a sign of **good model calibration**, not penalized. Only excessive hedging (>4 indicators) with text-heuristic confidence triggers a minor adjustment. This reflects clinical best practice where appropriate uncertainty expression is desirable.

### Critical Finding Detection

Based on ACR Critical Results guidelines: pneumothorax, aortic dissection, pulmonary embolism, acute stroke, active hemorrhage, cardiac tamponade, spinal cord compression, and more. Any critical pattern triggers **IMMEDIATE referral** regardless of confidence score.

## Project Structure

```
voicerad/
  backend/
    app.py              # FastAPI server v1.3.0
    safety.py           # Clinical safety engine v1.3.0
    benchmarks.py       # CheXpert benchmark runner
    requirements.txt
    Dockerfile
    models/
      medgemma_wrapper.py  # MedGemma 4B with log-prob confidence
      medasr_wrapper.py    # Whisper STT
      dicom_utils.py       # DICOM handling
  frontend/
    src/
      App.jsx           # Main app with safety integration
      components/
        SafetyBanner.jsx  # Safety rail visual component
  benchmarks/
    clinical_validation.json  # 20 CheXpert-labeled test cases
  tests/
    test_safety.py      # 25 safety rail unit tests
    test_app.py         # API endpoint tests
  docker-compose.yml    # CPU + GPU profiles
  .github/workflows/    # CI/CD
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- (Optional) NVIDIA GPU with CUDA 12.1 for real model inference

### Backend
```bash
cd backend
pip install -r requirements.txt

# Demo mode (no GPU needed)
DEMO_MODE=true python app.py

# Real mode (requires HuggingFace token with MedGemma access)
HF_TOKEN=your_token python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker
```bash
# CPU demo mode
docker compose up

# GPU mode with real models
docker compose --profile gpu up
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | System status, model availability |
| POST | /api/voice/transcribe | Speech-to-text (Whisper) |
| POST | /api/images/upload | Image upload with DICOM detection |
| POST | /api/interpret/start-session | Start interpretation with safety |
| POST | /api/interpret/continue/{id} | Refine with additional context |
| POST | /api/interpret/finalize/{id} | Generate final report |
| POST | /api/safety/assess | Standalone safety assessment |
| POST | /api/benchmarks/run | Run clinical validation |
| POST | /api/tts/speak | Text-to-speech via edge-tts |

## Safety Philosophy

VoiceRad follows a **fail-safe design principle**:

1. The AI can never autonomously diagnose. Every output requires clinician review.
2. Safety overrides are deterministic. No probabilistic model can override the safety engine.
3. Confidence below 30% blocks output entirely.
4. Critical findings trigger immediate referral.
5. Safety is monotonically increasing. Additional information can only make the system more cautious.

**This is not a replacement for radiologists.** It is a tool to help clinicians in underserved areas where no radiologist is available, with built-in guardrails that protect patients when the AI is uncertain.

## License

Apache 2.0

## Team

Built by [Naivedhya Agrawal](https://www.kaggle.com/naivedhyaagrawal) for the Kaggle MedGemma Impact Challenge 2026.
# VoiceRad

**Voice-Controlled Mobile Radiology Assistant with Clinical Safety Rails**

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) using Google's MedGemma and Health AI Developer Foundations ([HAI-DEF](https://github.com/Google-Health/hai-def)).

---

## The Problem

Over **42 000 radiologist positions** will be unfilled globally by 2033, and **4.5 billion people** lack access to basic medical imaging services (WHO). Rural clinicians often have no radiologist within hours of travel.

## The Solution

VoiceRad is a **voice-first, offline-capable PWA** that lets rural clinicians upload a chest X-ray, ask questions by voice, and receive AI-assisted interpretations powered by **MedGemma 1.5 4B** with built-in clinical safety rails.

### What makes VoiceRad different

1. **Voice-first design** for hands-free clinical use (voice input via Whisper STT, audio output via edge-tts)
2. **Clinical safety rails** that classify every interpretation by triage level (CRITICAL / URGENT / ROUTINE / NORMAL), confidence score, and referral triggers
3. **Multi-turn agentic conversation** where clinicians can refine interpretations with additional clinical context
4. **Offline-first architecture** with IndexedDB queueing and background sync
5. **DICOM native support** with metadata extraction and conversion

---

## Architecture

```
[Voice Input] --> [Whisper STT] --> [FastAPI Backend]
[Image Upload] --> [DICOM/PNG/JPG] ---|
                                       v
                               [MedGemma 1.5 4B]
                               (4-bit NF4 quantized)
                                       |
                                       v
                              [Clinical Safety Engine]
                              (triage + confidence + referral)
                                       |
                                       v
                              [React PWA Frontend]
                              (SafetyBanner + TTS playback)
```

### Backend (FastAPI + Python)

- `backend/app.py` — API server v1.3.0 with safety-integrated endpoints
- `backend/safety.py` — Deterministic clinical safety engine (triage, confidence thresholds, referral triggers)
- `backend/benchmarks.py` — CheXpert-aligned clinical benchmark runner
- `backend/models/medgemma_wrapper.py` — MedGemma 1.5 4B with 4-bit NF4 quantization
- `backend/models/medasr_wrapper.py` — Whisper-based medical speech recognition
- `backend/models/dicom_utils.py` — DICOM to PIL conversion and metadata extraction

### Frontend (React PWA)

- `frontend/src/App.jsx` — Main app with safety state tracking and server TTS
- `frontend/src/components/SafetyBanner.jsx` — Visual safety rail UI (triage colors, confidence bar, referral alerts)
- Offline-first with IndexedDB + service worker sync

---

## Clinical Safety Rails

Every AI interpretation passes through a deterministic safety engine that **cannot be bypassed by the model**. Safety is monotonically increasing — more information can only make the system MORE cautious, never less.

### Triage Classification

| Level | Color | Action Required |
|-------|-------|----------------|
| CRITICAL | Red (pulsing) | Immediate radiologist review |
| URGENT | Orange | Review within 1 hour |
| ROUTINE | Yellow | Standard review queue |
| NORMAL | Green | No acute findings |

### Confidence Thresholds

| Range | Label | Behavior |
|-------|-------|----------|
| >= 85% | HIGH | Green indicator. Clinician sign-off still required. |
| 60-84% | MODERATE | Yellow indicator. Standard review. |
| 30-59% | LOW | Orange indicator. Prominent warning banner. |
| < 30% | VERY_LOW | **BLOCKED.** Interpretation withheld. Forced referral. |

### Critical Finding Detection

The safety engine scans every interpretation for ACR Critical Results patterns:

- Pneumothorax (including tension)
- Aortic dissection / rupture
- Pulmonary embolism
- Acute stroke / infarct
- Active hemorrhage
- Cardiac tamponade
- Spinal cord compression
- Necrotizing fasciitis
- And more...

Any critical pattern triggers **IMMEDIATE referral** regardless of confidence score.

### Referral Triggers

- **IMMEDIATE**: Critical findings detected, or model confidence < 30%
- **URGENT**: Urgent findings (suspected malignancy, large effusions, bowel obstruction, DVT)
- **Hedging penalty**: If the model uses uncertain language ("cannot determine", "possible", "limited study"), confidence is automatically reduced

### Human-in-Loop Flagging

**Every single interpretation** is flagged with "Clinician review required — AI-assisted only", regardless of confidence level. VoiceRad is designed as a clinical decision SUPPORT tool, never an autonomous diagnostic system.

---

## Clinical Benchmarks

VoiceRad includes a built-in benchmark runner aligned with **CheXpert pathology labels** for clinical validation.

### Benchmark Cases (`benchmarks/clinical_validation.json`)

20 CheXpert-labeled test cases covering:
- Normal chest X-rays
- Cardiomegaly
- Pleural effusion
- Consolidation / Pneumonia
- Pneumothorax (including tension)
- Atelectasis
- Lung mass / suspected malignancy
- Pulmonary edema
- Multi-pathology combinations

### Metrics Tracked

- **Per-pathology sensitivity and specificity** (Cardiomegaly, Effusion, Consolidation, Pneumothorax, Atelectasis)
- **Safety triage accuracy** (does the system correctly classify CRITICAL vs ROUTINE?)
- **Critical finding detection rate** (does the system catch pneumothorax / PE / aortic dissection?)
- **Average confidence scores** per triage level

### Running Benchmarks

```bash
# Via API
curl -X POST http://localhost:8000/api/benchmarks/run

# Programmatic
from benchmarks import BenchmarkRunner
runner = BenchmarkRunner(model=medgemma_model, safety_engine=safety_engine)
summary = runner.run_suite(test_cases)
print(summary.to_dict())
```

---

## Performance Optimizations

| Metric | Before | After |
|--------|--------|-------|
| Inference speed (T4 GPU) | 21-35 seconds | ~7-10 seconds |
| VRAM usage | ~8 GB (float16) | ~3 GB (4-bit NF4) |
| DICOM on refine turn 2+ | Crash | Fixed |
| STT model loading | Fails (fake model) then fallback | Direct Whisper load |
| Server TTS | Stub (no-op) | Real audio via edge-tts |
| GPU Docker support | None | Full NVIDIA runtime |
| Offline sync | Always returns 0 | Real queue tracking |

### Key Optimizations

- **4-bit NF4 quantization** via BitsAndBytesConfig — cuts VRAM by 60%, improves speed 3x
- **Greedy decoding** (do_sample=False) for deterministic, faster clinical output
- **GPU memory cleanup** with gc.collect() + torch.cuda.empty_cache() in try/finally blocks
- **PIL image caching** in session — no more re-encoding the same image on every refine turn
- **Direct Whisper loading** — removed fake google/medASR attempt that always failed

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) NVIDIA GPU with CUDA 12.1 for real model inference

### Backend

```bash
cd backend
pip install -r requirements.txt
# Demo mode (no GPU needed)
DEMO_MODE=true python app.py
# Real mode (requires HuggingFace token with MedGemma access)
HF_TOKEN=your_token python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
# CPU demo mode
docker compose up

# GPU mode with real models
docker compose --profile gpu up
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | System status, model availability, active sessions |
| POST | `/api/voice/transcribe` | Speech-to-text (Whisper) |
| POST | `/api/images/upload` | Image upload with DICOM detection |
| POST | `/api/images/convert-dicom` | DICOM to PNG conversion |
| POST | `/api/interpret/start-session` | Start interpretation session (returns safety assessment) |
| POST | `/api/interpret/continue/{id}` | Refine with additional context (returns updated safety) |
| POST | `/api/interpret/finalize/{id}` | Generate final report with safety history |
| POST | `/api/safety/assess` | Standalone safety assessment for any clinical text |
| POST | `/api/benchmarks/run` | Run clinical validation benchmarks |
| POST | `/api/tts/speak` | Text-to-speech via edge-tts |
| POST | `/api/sync/queue` | Queue offline requests |
| GET | `/api/sync/status` | Check sync queue status |

---

## Project Structure

```
voicerad/
  backend/
    app.py                    # FastAPI server v1.3.0
    safety.py                 # Clinical safety engine
    benchmarks.py             # CheXpert benchmark runner
    requirements.txt          # Python dependencies
    Dockerfile                # CPU/GPU via build arg
    models/
      medgemma_wrapper.py     # MedGemma 4-bit quantized inference
      medasr_wrapper.py       # Whisper STT
      dicom_utils.py          # DICOM handling
  frontend/
    src/
      App.jsx                 # Main app with safety integration
      App.css                 # Styles including safety animations
      components/
        SafetyBanner.jsx      # Safety rail visual component
  benchmarks/
    clinical_validation.json  # 20 CheXpert-labeled test cases
  tests/
    test_safety.py            # 25 safety rail unit tests
    test_app.py               # API endpoint tests
  docker-compose.yml          # CPU + GPU profiles
  .github/workflows/          # CI/CD
```

---

## Safety Philosophy

VoiceRad follows a **fail-safe** design principle:

1. **The AI can never autonomously diagnose.** Every output requires clinician review.
2. **Safety overrides are deterministic.** No probabilistic model can override the safety engine.
3. **Confidence below 30% blocks output entirely.** The system would rather show nothing than show something dangerously wrong.
4. **Critical findings trigger immediate referral.** Pneumothorax, PE, aortic dissection — these are never treated as routine.
5. **Safety is monotonically increasing.** Additional information or hedging can only make the system more cautious.

This is not a replacement for radiologists. It is a tool to help clinicians in underserved areas where no radiologist is available, with built-in guardrails that protect patients when the AI is uncertain.

---

## License

Apache 2.0

## Team

Built for the Kaggle MedGemma Impact Challenge 2026.
