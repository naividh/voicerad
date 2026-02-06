# VoiceRad — Kaggle MedGemma Impact Challenge Writeup

> **Team:** naividh  
> **Repository:** [github.com/naividh/voicerad](https://github.com/naividh/voicerad)  
> **License:** Apache 2.0  
> **Video demo:** *(link to be added)*  

---

## 1. Problem & Motivation

The global shortage of radiologists is acute and worsening. An estimated 42,000 radiologist positions will remain unfilled by 2033 (Royal College of Radiologists, 2023). The World Health Organization reports that 4.5 billion people — more than half the planet — lack access to basic medical imaging interpretation. The burden falls hardest on rural and low-resource settings, where a single general-practice clinician handles all imaging without specialist support, often with unreliable internet and no desktop workstation.

These clinicians need a tool that meets them where they are: on a mobile phone, with voice interaction (because their hands are occupied), working even when connectivity drops, and powered by AI that understands medical images at a specialist level.

No existing product combines voice-controlled interaction, multimodal medical image analysis, offline operation, and agentic multi-turn refinement in a single mobile-ready application. VoiceRad fills that gap.

---

## 2. Solution Overview

VoiceRad is a Progressive Web App (PWA) that enables any clinician with a smartphone to:

1. **Speak** a clinical question ("What do you see in this chest X-ray?")
2. **Upload** a medical image from the phone camera or file system
3. **Receive** an AI-powered radiology interpretation with voice playback
4. **Refine** the interpretation through multi-turn agentic conversation
5. **Generate** a structured radiology report with mandatory clinician-review disclaimer
6. **Work offline** via Service Worker caching, syncing results when connectivity returns

The backend is a single FastAPI server running MedGemma 1.5 4B for multimodal image interpretation and Whisper for speech-to-text. The frontend is a React 18 SPA served as an installable PWA. The entire stack is containerized with Docker Compose for one-command deployment.

---

## 3. Effective Use of HAI-DEF Models (20%)

VoiceRad integrates three models from Google's Health AI Developer Foundations (HAI-DEF):

**MedGemma 1.5 4B-IT** is the core interpretation engine. It receives the clinician's spoken question alongside the uploaded medical image and produces structured radiology findings, impressions, and follow-up suggestions. Critically, MedGemma is used in a multi-turn loop: after the initial interpretation, the clinician can provide additional clinical context (symptoms, history, suspected diagnosis), and MedGemma refines its analysis incorporating the full conversation history. This goes well beyond single-shot classification — it replicates the iterative reasoning process a radiologist uses during a real read.

**MedASR** (google/medASR) is the intended speech-to-text model for medical terminology. The wrapper attempts to load MedASR first; if unavailable, it falls back gracefully to OpenAI Whisper (base), ensuring the voice pipeline never breaks. This design means VoiceRad is ready to adopt MedASR the moment it is publicly released, while remaining fully functional today.

**MedSigLIP** is integrated in the architecture for visual feature extraction and image preprocessing, complementing MedGemma's multimodal capabilities.

All model loading is gated behind a DEMO_MODE environment variable, allowing the full application to run without GPU for development, testing, and demonstration purposes. On a Kaggle T4 GPU (16 GB), MedGemma loads and runs inference in under 10 seconds.

---

## 4. Impact Potential (15%)

The target users are general-practice clinicians in rural and low-resource settings — the exact population most underserved by current radiology infrastructure. VoiceRad's impact is direct and measurable:

- **Access:** Provides specialist-level radiology interpretation where no radiologist exists within hours of travel
- **Efficiency:** Voice interaction means clinicians can query the AI during a patient exam without pausing to type
- **Safety:** Every output carries a mandatory disclaimer and a "requires clinician review" flag — the AI assists, it does not diagnose
- **Equity:** As a PWA, it runs on any device with a browser — no app store, no specialized hardware, no license fees
- **Offline-first:** The Service Worker caches the application shell; pending interpretations queue locally and sync when connectivity returns

The offline-first design is not a nice-to-have; it is essential. In rural sub-Saharan Africa, South Asia, and Latin America, mobile internet is intermittent. A tool that requires constant connectivity is useless in the settings where it is needed most.

---

## 5. Product Feasibility (20%)

VoiceRad is a fully functional, deployable application — not a notebook or prototype. The codebase includes:

- **Backend:** FastAPI with session management (TTL, cleanup, max cap), input validation (422 for empty files, 413 for oversized images), configurable CORS, structured error handlers, and a clean lifespan lifecycle for model loading/unloading
- **Frontend:** React 18 with component-split architecture (useVoiceRecorder hook, ImageUpload, InterpretationView), responsive CSS, and PWA manifest with Service Worker
- **Testing:** 26 pytest tests covering every API endpoint, edge cases (empty files, missing sessions, session expiry, capacity limits), and the full multi-turn workflow — all runnable in demo mode without GPU
- **Deployment:** Single `docker-compose up` brings up the entire stack. Environment variables control CORS origins, demo mode, and port
- **Documentation:** Comprehensive README with API reference, architecture diagram, environment variables, quick start, test instructions, and model fallback documentation

The codebase follows production patterns: session TTL prevents memory leaks, input validation prevents 500 errors, CORS is locked to configured origins, and the frontend is split into reusable components. It is not enterprise-grade (it lacks authentication, rate limiting per-IP, and database-backed sessions), but it is honest about those gaps and structured so they can be added incrementally.

---

## 6. Execution & Communication (30%)

**Code quality:** The repository is public, Apache 2.0 licensed, with 25+ commits showing iterative development. Variable names are descriptive, functions are documented, and the code is organized into clear modules (models/, frontend/src/, tests/).

**Testing:** The test suite covers happy paths and error paths for all endpoints. Tests run in demo mode (no GPU) with a single command: `DEMO_MODE=1 pytest`.

**Documentation:** The README serves as both a user guide and a developer guide, with a quick start, API table, architecture overview, model documentation, environment variable reference, and project structure map.

**Video:** The 3-minute demo walks through the complete clinician workflow — voice input, image upload, AI interpretation, multi-turn refinement, final report — followed by architecture and prize alignment.

**Honest limitations:** VoiceRad does not claim to be production-ready for autonomous diagnosis. It is a demonstration of how HAI-DEF models can be composed into a clinically meaningful, accessible application. The MedASR model ID is documented as speculative with a working Whisper fallback. The disclaimer is prominent and non-negotiable.

---

## 7. Prize Alignment

| Prize | Qualification |
|-------|---------------|
| **Main Track ($75K)** | Complete application addressing a documented clinical gap with real-world deployment path |
| **Edge AI ($5K)** | MedGemma 4B fits on a T4 GPU; the PWA runs on mobile hardware; offline-first architecture |
| **Agentic Workflow ($10K)** | Voice → Image → Interpretation → Multi-turn refinement loop with full conversation history |
| **Novel Task ($10K)** | Structured radiology report generation from multimodal agentic conversation — not classification |

---

## References

1. Royal College of Radiologists. *Clinical Radiology Workforce Census Report*, 2023.
2. World Health Organization. *Global Atlas of Medical Devices*, 2022.
3. Google Health AI Developer Foundations. [github.com/Google-Health/hai-def](https://github.com/Google-Health/hai-def)
4. MedGemma 1.5 4B-IT. [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
