"""
VoiceRad - Voice-Controlled Mobile Radiology Assistant
FastAPI Backend Server (v1.3.0)

v1.3.0 additions:
- Clinical safety rails integrated into all interpretation endpoints
- Safety assessment (triage, confidence, referral triggers) on every response
- /api/safety/assess endpoint for standalone safety evaluation
- /api/benchmarks/run endpoint for clinical validation
- Structured interpretation response with safety metadata
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
import logging
import os
import sys
import uuid
import time
import hashlib
import random
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -- Configuration -------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_IMAGE_SIZE = 50_000_000
MAX_SESSIONS = 100
SESSION_TTL_SECONDS = 1800
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")


class AppState:
    """Holds runtime state for the application."""
    def __init__(self):
        self.medgemma_model = None
        self.medasr_model = None
        self.safety_engine = None
        self.device = self._detect_device()
        self.sessions: dict = {}
        self._rate_limits: dict = defaultdict(list)
        self._sync_queue: deque = deque(maxlen=500)

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def create_session(self, pil_image, image_bytes, question):
        self._cleanup_expired_sessions()
        if len(self.sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "Too many active sessions.")
        session_id = str(uuid.uuid4())[:12]
        self.sessions[session_id] = {
            "created": time.time(),
            "pil_image": pil_image,
            "image_bytes": image_bytes,
            "question": question,
            "turns": [],
            "safety_history": [],
        }
        return session_id

    def get_session(self, session_id):
        self._cleanup_expired_sessions()
        if session_id not in self.sessions:
            raise HTTPException(404, "Session not found")
        return self.sessions[session_id]

    def pop_session(self, session_id):
        self._cleanup_expired_sessions()
        if session_id not in self.sessions:
            raise HTTPException(404, "Session not found")
        return self.sessions.pop(session_id)

    def _cleanup_expired_sessions(self):
        now = time.time()
        expired = [
            sid for sid, d in self.sessions.items()
            if now - d["created"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.sessions.pop(sid, None)

    def check_rate_limit(self, client_ip):
        if DEMO_MODE:
            return
        now = time.time()
        window = [t for t in self._rate_limits[client_ip] if now - t < 60]
        self._rate_limits[client_ip] = window
        if len(window) >= RATE_LIMIT_RPM:
            raise HTTPException(429, "Rate limit exceeded.")
        self._rate_limits[client_ip].append(now)


app_state = AppState()


# -- Lifespan -----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VOICERAD BACKEND v1.3.0 STARTING")
    logger.info("Device: %s | Demo: %s", app_state.device, DEMO_MODE)

    # Always load safety engine (it's deterministic, no GPU needed)
    from safety import ClinicalSafetyEngine
    app_state.safety_engine = ClinicalSafetyEngine()
    logger.info("Clinical safety engine loaded")

    if not DEMO_MODE:
        try:
            from models.medgemma_wrapper import MedGemmaModel
            from models.medasr_wrapper import MedASRModel
            app_state.medgemma_model = MedGemmaModel(device=app_state.device)
            app_state.medasr_model = MedASRModel(device=app_state.device)
            logger.info("All models loaded successfully")
        except Exception as exc:
            logger.warning("Models unavailable: %s", exc)
    else:
        logger.info("DEMO_MODE -- skipping model loading")

    logger.info("BACKEND READY http://localhost:8000")
    logger.info("=" * 60)
    yield
    app_state.medgemma_model = app_state.medasr_model = None


app = FastAPI(
    title="VoiceRad API",
    description="Voice-Controlled Radiology Assistant with Clinical Safety Rails",
    version="1.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s (%.0fms)",
        request.method, request.url.path,
        response.status_code, (time.time() - start) * 1000,
    )
    return response


def _bytes_to_pil(image_bytes):
    from models.dicom_utils import is_dicom, dicom_to_pil
    from PIL import Image as PILImage
    import io
    if is_dicom(image_bytes):
        return dicom_to_pil(image_bytes)
    return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")


def _run_safety(interpretation: str) -> dict:
    """Run safety assessment and return dict. Always works even without engine."""
    if app_state.safety_engine:
        return app_state.safety_engine.assess(interpretation).to_dict()
    return {
        "triage_level": "ROUTINE",
        "confidence_score": 0.5,
        "confidence_label": "MODERATE",
        "requires_human_review": True,
        "is_blocked": False,
        "block_reason": "",
        "referral_triggered": False,
        "referral_type": "",
        "referral_reason": "",
        "critical_findings": [],
        "urgent_findings": [],
        "hedging_indicators": [],
        "warnings": ["Safety engine not loaded. Treat all output with caution."],
    }


# -- Routes -------------------------------------------------

@app.get("/")
async def root():
    return {"name": "VoiceRad API", "version": "1.3.0", "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.3.0",
        "device": app_state.device,
        "demo_mode": DEMO_MODE or app_state.medgemma_model is None,
        "models": {
            "medgemma": app_state.medgemma_model is not None,
            "medasr": app_state.medasr_model is not None,
            "safety_engine": app_state.safety_engine is not None,
        },
        "active_sessions": len(app_state.sessions),
    }


@app.post("/api/voice/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    app_state.check_rate_limit(request.client.host)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(422, "Empty audio file")
    if app_state.medasr_model:
        transcript = app_state.medasr_model.transcribe(audio_bytes)
    else:
        transcript = random.choice([
            "What do you see in this chest X-ray?",
            "Any signs of pneumonia?",
            "Check for fractures",
            "Describe the findings",
        ])
    return {"transcript": transcript, "confidence": 0.95}


@app.post("/api/images/upload")
async def upload_image(request: Request, image: UploadFile = File(...)):
    app_state.check_rate_limit(request.client.host)
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty image file")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(413, "Image too large")
    from models.dicom_utils import is_dicom, extract_dicom_metadata
    metadata = {}
    if is_dicom(image_bytes):
        try:
            metadata = extract_dicom_metadata(image_bytes)
        except Exception:
            pass
    return {
        "image_id": "img_" + hashlib.sha256(image_bytes).hexdigest()[:12],
        "filename": image.filename,
        "size_bytes": len(image_bytes),
        "is_dicom": is_dicom(image_bytes),
        "dicom_metadata": metadata or None,
    }


# -- Interpretation with Safety Rails -----------------------

@app.post("/api/interpret/start-session")
async def start_session(
    request: Request,
    image: UploadFile = File(...),
    question: Optional[str] = Form(None),
):
    app_state.check_rate_limit(request.client.host)
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty image file")

    pil_image = _bytes_to_pil(image_bytes)
    session_id = app_state.create_session(pil_image, image_bytes, question)

    if app_state.medgemma_model:
        interpretation = app_state.medgemma_model.interpret(
            pil_image, question or "Describe findings in this medical image."
        )
    else:
        interpretation = _demo_interpretation(question)

    # Run safety assessment on interpretation
    safety = _run_safety(interpretation)

    # Store safety in session history
    session = app_state.get_session(session_id)
    session["safety_history"].append(safety)

    # If safety blocks the interpretation, replace with referral message
    display_interpretation = interpretation
    if safety.get("is_blocked"):
        display_interpretation = (
            "AI INTERPRETATION WITHHELD\n"
            f"Reason: {safety.get('block_reason', 'Low confidence')}\n\n"
            "This image requires human radiologist review.\n"
            "The AI system's confidence is below the safety threshold."
        )

    return {
        "session_id": session_id,
        "status": "session_started",
        "interpretation": display_interpretation,
        "raw_interpretation": interpretation if not safety.get("is_blocked") else None,
        "safety": safety,
        "clarifying_questions": _get_contextual_questions(safety),
        "requires_review": True,
    }


@app.post("/api/interpret/continue/{session_id}")
async def continue_session(
    session_id: str,
    answer: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
):
    session = app_state.get_session(session_id)

    if audio and not answer:
        audio_bytes = await audio.read()
        answer = (
            app_state.medasr_model.transcribe(audio_bytes)
            if app_state.medasr_model
            else "(voice input -- demo mode)"
        )
    if not answer:
        raise HTTPException(422, "Provide text or audio")

    session["turns"].append({"input": answer, "ts": time.time()})

    if app_state.medgemma_model:
        refined = app_state.medgemma_model.refine(
            session["pil_image"], answer, session["turns"]
        )
    else:
        refined = _demo_refine(answer, len(session["turns"]))

    safety = _run_safety(refined)
    session["safety_history"].append(safety)

    display = refined
    if safety.get("is_blocked"):
        display = (
            "REFINED INTERPRETATION WITHHELD\n"
            f"Reason: {safety.get('block_reason')}\n"
            "Refer to radiologist."
        )

    return {
        "session_id": session_id,
        "refined_interpretation": display,
        "safety": safety,
        "turn": len(session["turns"]),
    }


@app.post("/api/interpret/finalize/{session_id}")
async def finalize(session_id: str):
    session = app_state.pop_session(session_id)
    report = _build_final_report(session_id, session)
    safety = _run_safety(report)

    return {
        "session_id": session_id,
        "status": "completed",
        "final_report": report,
        "safety": safety,
        "safety_history": session.get("safety_history", []),
        "total_turns": len(session["turns"]),
        "requires_clinician_review": True,
    }


# -- Safety endpoint ----------------------------------------

@app.post("/api/safety/assess")
async def safety_assess(text: str = Form(...)):
    """Standalone safety assessment for any clinical text."""
    if not text or not text.strip():
        raise HTTPException(422, "Empty text")
    safety = _run_safety(text.strip())
    return {"text_length": len(text.strip()), "safety": safety}


# -- Benchmark endpoint -------------------------------------

@app.post("/api/benchmarks/run")
async def run_benchmarks():
    """Run clinical benchmarks against loaded model."""
    from benchmarks import BenchmarkRunner, BenchmarkCase

    runner = BenchmarkRunner(
        model=app_state.medgemma_model,
        safety_engine=app_state.safety_engine,
    )

    # Built-in test cases (no images needed for safety/NLP testing)
    test_cases = [
        BenchmarkCase(
            case_id="DEMO-001",
            question="Describe this chest X-ray",
            ground_truth_labels=["No Finding"],
            expected_triage="NORMAL",
            description="Normal CXR baseline",
        ),
        BenchmarkCase(
            case_id="DEMO-002",
            question="Check for pneumonia",
            ground_truth_labels=["Consolidation", "Pneumonia"],
            expected_triage="URGENT",
            description="Lobar pneumonia case",
        ),
        BenchmarkCase(
            case_id="DEMO-003",
            question="Evaluate cardiac silhouette",
            ground_truth_labels=["Cardiomegaly", "Pleural Effusion"],
            expected_triage="ROUTINE",
            description="CHF case",
        ),
    ]

    summary = runner.run_suite(test_cases)
    return {"benchmark_results": summary.to_dict()}


# -- TTS endpoint -------------------------------------------

@app.post("/api/tts/speak")
async def tts_speak(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(422, "Empty text")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text.strip(), "en-US-AriaNeural")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        if audio_bytes:
            return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
    return {"status": "use_client_tts", "text": text.strip()}


# -- DICOM conversion ----------------------------------------

@app.post("/api/images/convert-dicom")
async def convert_dicom(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty file")
    from models.dicom_utils import is_dicom, dicom_to_pil, extract_dicom_metadata
    import io, base64
    if not is_dicom(image_bytes):
        raise HTTPException(400, "Not a DICOM file")
    pil_image = dicom_to_pil(image_bytes)
    metadata = extract_dicom_metadata(image_bytes)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return {
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
        "metadata": metadata,
        "width": pil_image.width,
        "height": pil_image.height,
    }


# -- Sync endpoints -----------------------------------------

@app.post("/api/sync/queue")
async def sync_queue(
    request: Request,
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
):
    entry = {
        "id": str(uuid.uuid4())[:8],
        "question": question,
        "has_image": image is not None,
        "ts": time.time(),
        "status": "pending",
    }
    app_state._sync_queue.append(entry)
    return {"status": "queued", "id": entry["id"], "pending": len(app_state._sync_queue)}


@app.get("/api/sync/status")
async def sync_status():
    pending = sum(1 for e in app_state._sync_queue if e["status"] == "pending")
    return {"pending": pending, "total": len(app_state._sync_queue)}


# -- Contextual question generation --------------------------

def _get_contextual_questions(safety: dict) -> list:
    """Generate clinically relevant follow-up questions based on safety assessment."""
    questions = []

    triage = safety.get("triage_level", "ROUTINE")
    if triage == "CRITICAL":
        questions = [
            "Is the patient hemodynamically stable?",
            "Is this a new finding or known condition?",
            "What are the current vital signs?",
        ]
    elif triage == "URGENT":
        questions = [
            "What are the patient's symptoms and duration?",
            "Any relevant surgical or medical history?",
            "Is there comparison imaging available?",
        ]
    else:
        questions = [
            "Any specific symptoms prompting this study?",
            "Previous imaging available for comparison?",
            "Suspected diagnosis or clinical concern?",
        ]

    if safety.get("confidence_label") in ("LOW", "VERY_LOW"):
        questions.insert(0, "Image quality concern detected. Can you provide a clearer image?")

    return questions


# -- Demo helpers -------------------------------------------

_DEMO_FINDINGS = [
    {
        "finding": "Bilateral patchy opacities in lower lobes",
        "impression": "Findings suggestive of bilateral lower lobe pneumonia. "
        "Clinical correlation with lab values (WBC, CRP) recommended.",
    },
    {
        "finding": "Heart size within normal limits. Clear lung fields bilaterally. "
        "No pleural effusion or pneumothorax",
        "impression": "No acute cardiopulmonary abnormality identified.",
    },
    {
        "finding": "Mild cardiomegaly. Small bilateral pleural effusions. "
        "Cephalization of pulmonary vasculature",
        "impression": "Findings consistent with mild congestive heart failure. "
        "Recommend clinical correlation and follow-up.",
    },
    {
        "finding": "Right upper lobe consolidation with air bronchograms. "
        "No significant mediastinal shift",
        "impression": "Right upper lobe consolidation likely infectious. "
        "Recommend sputum culture and follow-up imaging in 4-6 weeks.",
    },
    {
        "finding": "Large left-sided tension pneumothorax with mediastinal shift "
        "to the right. Complete collapse of the left lung",
        "impression": "CRITICAL: Tension pneumothorax requiring immediate "
        "needle decompression and chest tube placement.",
    },
]


def _demo_interpretation(question):
    q = question or "General evaluation"
    demo = random.choice(_DEMO_FINDINGS)
    return (
        f"RADIOLOGY INTERPRETATION (Demo)\n"
        f"{'=' * 50}\n"
        f"Question: {q}\n\n"
        f"FINDINGS:\n- {demo['finding']}\n\n"
        f"IMPRESSION:\n{demo['impression']}\n\n"
        f"NOTE: Demo response. Real MedGemma model not loaded."
    )


def _demo_refine(answer, turn_number):
    return (
        f"REFINED INTERPRETATION (Turn {turn_number})\n"
        f"{'=' * 50}\n"
        f"Additional context: {answer}\n\n"
        f"Assessment updated with clinical context.\n"
        f"NOTE: Demo response."
    )


def _build_final_report(session_id, session):
    turn_lines = "\n".join(
        f"  Turn {i+1}: {t['input']}" for i, t in enumerate(session["turns"])
    )
    return (
        f"FINAL RADIOLOGY REPORT\n"
        f"Session: {session_id}\n"
        f"Turns: {len(session['turns'])}\n"
        f"{turn_lines}\n\n"
        f"RECOMMENDATIONS:\n"
        f"1. Clinical correlation essential\n"
        f"2. Radiologist review required\n\n"
        f"DISCLAIMER: AI-assisted. Clinician review mandatory."
    )


# -- Error handlers -----------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development",
    )
