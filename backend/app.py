"""
VoiceRad - Voice-Controlled Mobile Radiology Assistant
FastAPI Backend Server
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import os
import uuid
import time
import hashlib
import random
from contextlib import asynccontextmanager
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MAX_IMAGE_SIZE = 50_000_000  # 50 MB
MAX_SESSIONS = 100
SESSION_TTL_SECONDS = 1800  # 30 minutes
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")


class AppState:
    """Holds runtime state for the application."""

    def __init__(self):
        self.medgemma_model = None
        self.medasr_model = None
        self.device = self._detect_device()
        self.sessions: dict = {}

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ── Session management ───────────────────────────────
    def create_session(self, image_bytes: bytes, question: Optional[str]) -> str:
        """Create a new interpretation session with TTL enforcement."""
        self._cleanup_expired_sessions()
        if len(self.sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "Too many active sessions. Try again later.")
        session_id = str(uuid.uuid4())[:12]
        self.sessions[session_id] = {
            "created": time.time(),
            "image": image_bytes,
            "question": question,
            "turns": [],
        }
        return session_id

    def get_session(self, session_id: str) -> dict:
        """Retrieve a session or raise 404."""
        self._cleanup_expired_sessions()
        if session_id not in self.sessions:
            raise HTTPException(404, "Session not found")
        return self.sessions[session_id]

    def pop_session(self, session_id: str) -> dict:
        """Retrieve and remove a session or raise 404."""
        self._cleanup_expired_sessions()
        if session_id not in self.sessions:
            raise HTTPException(404, "Session not found")
        return self.sessions.pop(session_id)

    def _cleanup_expired_sessions(self):
        """Remove sessions older than SESSION_TTL_SECONDS."""
        now = time.time()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["created"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            logger.info("Expired session %s", sid)


app_state = AppState()


# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VOICERAD BACKEND STARTING")
    logger.info("Device: %s | Demo mode: %s", app_state.device, DEMO_MODE)

    if not DEMO_MODE:
        try:
            from models.medgemma_wrapper import MedGemmaModel
            from models.medasr_wrapper import MedASRModel
            app_state.medgemma_model = MedGemmaModel(device=app_state.device)
            app_state.medasr_model = MedASRModel(device=app_state.device)
            logger.info("All models loaded successfully")
        except Exception as exc:
            logger.warning("Models unavailable, falling back to demo: %s", exc)
    else:
        logger.info("DEMO_MODE enabled — skipping model loading")

    logger.info("BACKEND READY http://localhost:8000")
    logger.info("=" * 60)
    yield
    app_state.medgemma_model = app_state.medasr_model = None


# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="VoiceRad API",
    description="Voice-Controlled Radiology Assistant",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "VoiceRad API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "device": app_state.device,
        "demo_mode": DEMO_MODE or app_state.medgemma_model is None,
        "models": {
            "medgemma": app_state.medgemma_model is not None,
            "medasr": app_state.medasr_model is not None,
        },
        "active_sessions": len(app_state.sessions),
    }


@app.post("/api/voice/transcribe")
async def transcribe(audio: UploadFile = File(...)):
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
async def upload_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty image file")
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_SIZE // 1_000_000} MB limit")
    return {
        "image_id": "img_" + hashlib.sha256(image_bytes).hexdigest()[:12],
        "filename": image.filename,
        "size_bytes": len(image_bytes),
    }


@app.post("/api/interpret/start-session")
async def start_session(
    image: UploadFile = File(...),
    question: Optional[str] = Form(None),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty image file")

    session_id = app_state.create_session(image_bytes, question)

    if app_state.medgemma_model:
        from PIL import Image as PILImage
        import io
        pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        interpretation = app_state.medgemma_model.interpret(
            pil_image, question or "Describe findings in this medical image."
        )
    else:
        interpretation = _demo_interpretation(question)

    return {
        "session_id": session_id,
        "status": "session_started",
        "interpretation": interpretation,
        "clarifying_questions": [
            "Any specific symptoms?",
            "Previous imaging available?",
            "Suspected diagnosis?",
        ],
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
            else "(voice input — demo mode)"
        )

    if not answer:
        raise HTTPException(422, "Provide either text answer or audio")

    session["turns"].append({"input": answer, "ts": time.time()})

    if app_state.medgemma_model:
        refined = app_state.medgemma_model.refine(
            session["image"], answer, session["turns"]
        )
    else:
        refined = _demo_refine(answer, len(session["turns"]))

    return {
        "session_id": session_id,
        "refined_interpretation": refined,
        "turn": len(session["turns"]),
    }


@app.post("/api/interpret/finalize/{session_id}")
async def finalize(session_id: str):
    session = app_state.pop_session(session_id)
    return {
        "session_id": session_id,
        "status": "completed",
        "final_report": _build_final_report(session_id, session),
        "total_turns": len(session["turns"]),
        "requires_clinician_review": True,
    }


@app.post("/api/sync/queue")
async def sync_queue():
    return {"status": "queued"}


@app.get("/api/sync/status")
async def sync_status():
    return {"pending": 0}


# ── Demo helpers ─────────────────────────────────────────
def _demo_interpretation(question: Optional[str]) -> str:
    q = question or "General"
    return (
        f"RADIOLOGY INTERPRETATION (Demo)\n"
        f"{'=' * 50}\n"
        f"Question: {q}\n\n"
        f"FINDINGS:\n"
        f"- Image adequate for evaluation\n"
        f"- No acute abnormality identified\n"
        f"- Heart size normal\n"
        f"- No pleural effusion\n\n"
        f"IMPRESSION: No acute findings. Clinical correlation recommended."
    )


def _demo_refine(answer: str, turn_number: int) -> str:
    return (
        f"REFINED INTERPRETATION (Turn {turn_number})\n"
        f"Additional context: {answer}\n"
        f"Assessment updated with clinical context."
    )


def _build_final_report(session_id: str, session: dict) -> str:
    turn_lines = "\n".join(
        f"  Turn {i + 1}: {turn['input']}" for i, turn in enumerate(session["turns"])
    )
    return (
        f"FINAL RADIOLOGY REPORT\n"
        f"Session: {session_id}\n"
        f"Turns: {len(session['turns'])}\n"
        f"{turn_lines}\n\n"
        f"RECOMMENDATIONS:\n"
        f"1. Clinical correlation essential\n"
        f"2. Radiologist review required\n\n"
        f"DISCLAIMER: AI-assisted interpretation. Clinician review mandatory."
    )


# ── Error handlers ───────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ── Entrypoint ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development",
    )
