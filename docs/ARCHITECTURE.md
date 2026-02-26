# VoiceRad Architecture Documentation

> Technical deep-dive for competition judges and contributors.
> Last updated: February 2026 | Version 1.3.0

## System Overview

VoiceRad is a voice-controlled radiology assistant designed for resource-constrained
clinical environments. It combines MedGemma 4B (a medical vision-language model) with
Whisper-based speech recognition and clinical safety rails to enable hands-free
radiological interpretation.

### Design Philosophy

1. **Safety-first**: Every model output passes through clinical safety rails before
   reaching the clinician. The system can withhold interpretations below confidence
   thresholds.
2. **Voice-native**: Designed for gloved-hands, sterile environments where keyboard/mouse
   interaction is impractical.
3. **Progressive enhancement**: Works in demo mode without GPU, degrades gracefully when
   models are unavailable, and queues requests for offline-first operation.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  React PWA      │────│  FastAPI         │────│  MedGemma 4B    │
│  (Frontend)      │    │  (Backend)       │    │  (VLM Engine)   │
│                  │    │                  │    │                  │
│  - Voice Record  │    │  - Session Mgmt  │    │  - Image+Text    │
│  - Image Upload  │    │  - Rate Limiting  │    │  - 4-bit Quant   │
│  - Safety UI     │    │  - Safety Rails  │    │  - Log-prob Conf │
│  - Offline Queue │    │  - Benchmarks    │    │                  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        │                  ┌────┴────────────┐         │
        │                  │  Whisper STT    │         │
        │                  │  (MedASR)       │         │
        │                  └─────────────────┘         │
        │                                              │
        └──────────────────────────────────────────────┘
                    Clinical Safety Engine
                    (Deterministic, no GPU required)
```

## Component Details

### 1. MedGemma Wrapper (`backend/models/medgemma_wrapper.py`)

**Model**: `google/medgemma-4b-it` (4B parameter medical VLM)

**Quantization**: 4-bit (BitsAndBytes NF4) reducing VRAM from ~16GB to ~3GB

**Key methods**:
- `interpret(image, question)`: Single-turn image interpretation
- `interpret(image, question, return_confidence=True)`: Returns tuple of
  (text, confidence) where confidence is derived from model log-probabilities
- `refine(image, followup, history)`: Multi-turn refinement with conversation context
- `generate_report(image, history)`: Structured radiology report generation

**Confidence extraction**: When `return_confidence=True`, the wrapper computes
confidence from the mean log-probability of generated tokens:

```python
log_probs = outputs.scores  # per-token log-probabilities
mean_log_prob = torch.stack(log_probs).max(dim=-1).values.mean()
confidence = torch.sigmoid(mean_log_prob).item()
```

This gives a calibrated 0-1 score based on the model's actual certainty, unlike
text-heuristic approaches that guess confidence from word patterns.

### 2. Clinical Safety Engine (`backend/safety.py`)

The safety engine is **deterministic and GPU-free** -- it runs on every model output
regardless of deployment environment. Version 1.3.0 includes:

**Triage classification** (4 levels):
- NORMAL: No acute findings, routine follow-up
- ROUTINE: Non-urgent findings, standard workflow
- URGENT: Findings requiring attention within 24h
- CRITICAL: Immediate action required (e.g., tension pneumothorax)

**Confidence scoring** (dual-source):
1. **Model log-probabilities** (preferred): When the model provides calibrated
   confidence via `return_confidence=True`, the safety engine uses it directly.
   Source tracked as `model_logprob`.
2. **Text heuristic fallback**: When model confidence is unavailable, estimates
   from anatomical specificity, finding density, and clinical terminology.
   Source tracked as `text_heuristic`.

**Hedging language policy** (v1.3.0 change):
Previous versions penalized hedging language ("possible", "cannot exclude"),
reducing confidence scores. This was **clinically incorrect** -- appropriate
uncertainty language is desirable in radiology. v1.3.0 now:
- Awards a small bonus (+0.02 per indicator, max +0.10) for appropriate hedging
- Tracks hedging indicators in the safety response for transparency

**Referral triggers**: Pattern-matched critical findings that automatically
trigger specialist referral recommendations (e.g., aortic dissection -> vascular
surgery, tension pneumothorax -> emergency medicine).

### 3. Whisper STT (`backend/models/medasr_wrapper.py`)

Uses OpenAI Whisper (base model, ~74M parameters) for speech-to-text.

**Why Whisper, not a medical-specific ASR?**
There is no publicly available medical ASR model. The original codebase attempted
to load `google/medASR` which does not exist as a public HuggingFace model.
v1.2.0 fixed this to use Whisper directly with proper error handling.

**Audio pipeline**: Supports WAV, MP3, OGG, FLAC via soundfile/librosa with
automatic sample rate conversion to 16kHz mono.

### 4. Benchmark Framework (`backend/benchmarks.py`)

**CheXpert-style evaluation** with 14 pathology labels:

The benchmark runner evaluates model outputs against ground truth labels using
regex-based pathology extraction from free text. This is a known limitation --
structured output (JSON mode) would be more reliable, but MedGemma 4B's
instruction-tuned variant generates natural language reports.

**Metrics computed**:
- Per-pathology sensitivity, specificity, precision, F1
- Macro-averaged detection metrics
- Triage accuracy with over/under-escalation rates
- Safety rail effectiveness (critical finding detection rate)
- Confidence calibration (confidence when correct vs. when wrong)
- Latency percentiles (mean, median, P95)

**Validation data**: 20 CheXpert-labeled cases in
`benchmarks/clinical_validation.json` covering normal, routine, urgent, and
critical presentations.

### 5. React PWA Frontend (`frontend/`)

Progressive Web App with:
- Web Audio API integration for voice recording
- IndexedDB offline queue for disconnected operation
- Safety-aware UI: color-coded triage badges, confidence meters
- Responsive design for tablet/mobile use in clinical settings

## Session-Based Interpretation Flow

```
1. POST /api/interpret/start-session
   - Upload image + optional question
   - Returns: initial interpretation + safety assessment

2. POST /api/interpret/continue/{session_id}
   - Send follow-up text or audio
   - Returns: refined interpretation + updated safety

3. POST /api/interpret/finalize/{session_id}
   - Generate final structured report
   - Returns: report + full safety history across all turns
```

Each turn's safety assessment is stored in the session, enabling judges to see
how safety evaluations evolve as clinical context is added.

## Known Limitations & Honest Assessment

1. **Not truly agentic**: The current multi-turn flow is session-based (human-driven
   turns), not autonomous agent loops. The model does not self-select tools or decide
   when to re-analyze. This is a deliberate design choice for clinical safety -- 
   autonomous agent loops in medical contexts risk runaway hallucination.

2. **Pathology extraction is regex-based**: CheXpert label detection from free text
   uses keyword matching. A production system would use structured output (JSON mode)
   or a separate NER model.

3. **Single modality tested**: Only chest X-rays have been validated. CT, MRI, and
   ultrasound would require additional prompt engineering and validation.

4. **No fine-tuning**: MedGemma is used off-the-shelf. LoRA fine-tuning on 
   radiology-specific datasets (e.g., MIMIC-CXR) would likely improve performance.

5. **Edge deployment is aspirational**: The PWA works on mobile browsers, but
   MedGemma 4B inference requires a GPU server. True edge deployment would need
   model distillation or a smaller model variant.

## File Structure

```
voicerad/
├── backend/
│   ├── app.py                 # FastAPI server (v1.3.0)
│   ├── safety.py              # Clinical safety engine (v1.3.0)
│   ├── benchmarks.py          # CheXpert benchmark runner
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container
│   └── models/
│       ├── medgemma_wrapper.py # MedGemma 4B interface
│       ├── medasr_wrapper.py   # Whisper STT interface
│       └── dicom_utils.py      # DICOM parsing utilities
├── frontend/                      # React PWA
├── benchmarks/
│   └── clinical_validation.json # 20 CheXpert-labeled test cases
├── tests/
│   └── test_safety.py         # Safety engine test suite
├── docs/
│   └── ARCHITECTURE.md        # This file
└── .github/workflows/
    ├── ci.yml                 # CI pipeline
    └── firebase-deploy.yml    # Deployment pipeline
```

## Running Locally

```bash
# Backend (with GPU)
cd backend
pip install -r requirements.txt
DEMO_MODE=0 python app.py

# Backend (demo mode, no GPU)
DEMO_MODE=1 python app.py

# Frontend
cd frontend
npm install
npm start

# Tests
pytest tests/ -v

# Benchmarks (requires model loaded)
curl -X POST http://localhost:8000/api/benchmarks/run
```
