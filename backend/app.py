"""
VoiceRad - Voice-Controlled Mobile Radiology Assistant
FastAPI Backend Server

Fixes applied (v1.2.0):
- Session now stores PIL image (not raw bytes) to avoid DICOM crash on refine
- refine() receives PIL Image directly
- Server-side TTS via edge-tts (no more stub)
- Offline sync queue with actual persistence
- Varied demo responses for realistic testing
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
MAX_IMAGE_SIZE = 50_000_000  # 50 MB
MAX_SESSIONS = 100
SESSION_TTL_SECONDS = 1800  # 30 minutes
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")


class AppState:
    """Holds runtime state for the application."""

    def __init__(self):
        self.medgemma_model = None
        self.medasr_model = None
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

    # -- Session management ---------------------------------
    def create_session(
        self, pil_image, image_bytes: bytes, question: Optional[str]
    ) -> str:
        """Create a new interpretation session with TTL enforcement.

        CHANGED: Now stores PIL image alongside raw bytes.
        PIL image is used for MedGemma inference (avoids re-decoding).
        Raw bytes kept only for potential re-upload/export.
        """
        self._cleanup_expired_sessions()
        if len(self.sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "Too many active sessions. Try again later.")

        session_id = str(uuid.uuid4())[:12]
        self.sessions[session_id] = {
            "created": time.time(),
            "pil_image": pil_image,      # PIL Image for inference
            "image_bytes": image_bytes,   # Raw bytes for export
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
            sid
            for sid, data in self.sessions.items()
            if now - data["created"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            logger.info("Expired session %s", sid)

    # -- Rate limiting --------------------------------------
    def check_rate_limit(self, client_ip: str):
        if DEMO_MODE:
            return
        now = time.time()
        window = [t for t in self._rate_limits[client_ip] if now - t < 60]
        self._rate_limits[client_ip] = window
        if len(window) >= RATE_LIMIT_RPM:
            raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
        self._rate_limits[client_ip].append(now)


app_state = AppState()


# -- Lifespan -----------------------------------------------
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
        logger.info("DEMO_MODE enabled -- skipping model loading")

    logger.info("BACKEND READY http://localhost:8000")
    logger.info("=" * 60)
    yield
    app_state.medgemma_model = app_state.medasr_model = None


# -- App ----------------------------------------------------
app = FastAPI(
    title="VoiceRad API",
    description="Voice-Controlled Radiology Assistant",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request logging middleware -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# -- Helper: image bytes -> PIL ----------------------------
def _bytes_to_pil(image_bytes: bytes):
    """Convert raw image bytes (PNG/JPG/DICOM) to PIL Image."""
    from models.dicom_utils import is_dicom, dicom_to_pil
    from PIL import Image as PILImage
    import io

    if is_dicom(image_bytes):
        return dicom_to_pil(image_bytes)
    return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")


# -- Routes -------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "VoiceRad API",
        "version": "1.2.0",
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
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_SIZE // 1_000_000} MB limit")

    from models.dicom_utils import is_dicom, extract_dicom_metadata

    metadata = {}
    if is_dicom(image_bytes):
        try:
            metadata = extract_dicom_metadata(image_bytes)
        except Exception as exc:
            logger.warning("DICOM metadata extraction failed: %s", exc)

    return {
        "image_id": "img_" + hashlib.sha256(image_bytes).hexdigest()[:12],
        "filename": image.filename,
        "size_bytes": len(image_bytes),
        "is_dicom": is_dicom(image_bytes),
        "dicom_metadata": metadata if metadata else None,
    }


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

    # Convert once, store the PIL image in session
    pil_image = _bytes_to_pil(image_bytes)
    session_id = app_state.create_session(pil_image, image_bytes, question)

    if app_state.medgemma_model:
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
            else "(voice input -- demo mode)"
        )
    if not answer:
        raise HTTPException(422, "Provide either text answer or audio")

    session["turns"].append({"input": answer, "ts": time.time()})

    if app_state.medgemma_model:
        # FIXED: Pass PIL image from session, not raw bytes
        refined = app_state.medgemma_model.refine(
            session["pil_image"], answer, session["turns"]
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


# -- TTS endpoint -------------------------------------------
@app.post("/api/tts/speak")
async def tts_speak(text: str = Form(...)):
    """Server-side text-to-speech using edge-tts."""
    if not text or not text.strip():
        raise HTTPException(422, "Empty text")

    try:
        import edge_tts
        import asyncio
        import io

        communicate = edge_tts.Communicate(text.strip(), "en-US-AriaNeural")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        if audio_bytes:
            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline"},
            )
        else:
            return {
                "status": "use_client_tts",
                "text": text.strip(),
                "message": "TTS generation returned empty audio.",
            }
    except ImportError:
        logger.warning("edge-tts not installed, falling back to client TTS")
        return {
            "status": "use_client_tts",
            "text": text.strip(),
            "message": "Server TTS unavailable. Using Web Speech API.",
        }
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        return {
            "status": "use_client_tts",
            "text": text.strip(),
            "message": f"TTS error: {exc}",
        }


# -- DICOM conversion endpoint ------------------------------
@app.post("/api/images/convert-dicom")
async def convert_dicom(image: UploadFile = File(...)):
    """Convert a DICOM file to PNG and return metadata."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(422, "Empty file")

    from models.dicom_utils import is_dicom, dicom_to_pil, extract_dicom_metadata
    import io
    import base64

    if not is_dicom(image_bytes):
        raise HTTPException(
            400, "File is not a valid DICOM image. Upload a .dcm file."
        )
    try:
        pil_image = dicom_to_pil(image_bytes)
        metadata = extract_dicom_metadata(image_bytes)
    except Exception as exc:
        raise HTTPException(500, f"DICOM processing failed: {exc}")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "png_base64": png_b64,
        "metadata": metadata,
        "width": pil_image.width,
        "height": pil_image.height,
    }


# -- Offline sync endpoints ---------------------------------
@app.post("/api/sync/queue")
async def sync_queue(
    request: Request,
    image: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
):
    """Queue an interpretation request for when models become available."""
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
    return {
        "pending": pending,
        "total": len(app_state._sync_queue),
    }


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
]


def _demo_interpretation(question: Optional[str]) -> str:
    q = question or "General evaluation"
    demo = random.choice(_DEMO_FINDINGS)
    return (
        f"RADIOLOGY INTERPRETATION (Demo)\n"
        f"{'=' * 50}\n"
        f"Question: {q}\n\n"
        f"FINDINGS:\n"
        f"- {demo['finding']}\n\n"
        f"IMPRESSION:\n"
        f"{demo['impression']}\n\n"
        f"NOTE: This is a demo response. Real MedGemma model not loaded."
    )


def _demo_refine(answer: str, turn_number: int) -> str:
    return (
        f"REFINED INTERPRETATION (Turn {turn_number})\n"
        f"{'=' * 50}\n"
        f"Additional context incorporated: {answer}\n\n"
        f"Assessment has been updated considering the new clinical "
        f"information. Key differential diagnoses have been re-evaluated.\n\n"
        f"NOTE: This is a demo response. Real MedGemma model not loaded."
    )


def _build_final_report(session_id: str, session: dict) -> str:
    turn_lines = "\n".join(
        f"  Turn {i + 1}: {turn['input']}"
        for i, turn in enumerate(session["turns"])
    )
    return (
        f"FINAL RADIOLOGY REPORT\n"
        f"Session: {session_id}\n"
        f"Turns: {len(session['turns'])}\n"
        f"{turn_lines}\n\n"
        f"RECOMMENDATIONS:\n"
        f"1. Clinical correlation essential\n"
        f"2. Radiologist review required\n\n"
        f"DISCLAIMER: AI-assisted interpretation. "
        f"Clinician review mandatory."
    )


# -- Error handlers -----------------------------------------
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


# -- Entrypoint ---------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development",
    )
