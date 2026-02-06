"""
VoiceRad - Voice-Controlled Mobile Radiology Assistant
FastAPI Backend Server
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging, os, uuid, time, hashlib, random
from contextlib import asynccontextmanager
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.medgemma_model = None
        self.medasr_model = None
        self.device = self._get_device()
        self.sessions = {}

    def _get_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VOICERAD BACKEND STARTING")
    logger.info("Device: %s", app_state.device)
    try:
        from models.medgemma_wrapper import MedGemmaModel
        from models.medasr_wrapper import MedASRModel
        app_state.medgemma_model = MedGemmaModel(device=app_state.device)
        app_state.medasr_model = MedASRModel(device=app_state.device)
        logger.info("All models loaded")
    except Exception as e:
        logger.warning("Running in DEMO mode: %s", e)
    logger.info("BACKEND READY http://localhost:8000")
    logger.info("=" * 60)
    yield
    app_state.medgemma_model = app_state.medasr_model = None


app = FastAPI(title="VoiceRad API", description="Voice-Controlled Radiology Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"name": "VoiceRad API", "version": "1.0.0", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "device": app_state.device, "models": {"medgemma": app_state.medgemma_model is not None, "medasr": app_state.medasr_model is not None}, "active_sessions": len(app_state.sessions)}


@app.post("/api/voice/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    if app_state.medasr_model:
        t = app_state.medasr_model.transcribe(data)
    else:
        t = random.choice(["What do you see in this chest X-ray?", "Any signs of pneumonia?", "Check for fractures", "Describe the findings"])
    return {"transcript": t, "confidence": 0.95}


@app.post("/api/images/upload")
async def upload_image(image: UploadFile = File(...)):
    data = await image.read()
    if len(data) > 50_000_000:
        raise HTTPException(413, "Image exceeds 50 MB")
    return {"image_id": "img_" + hashlib.md5(data).hexdigest()[:12], "filename": image.filename, "size_bytes": len(data)}


@app.post("/api/interpret/start-session")
async def start_session(image: UploadFile = File(...), question: Optional[str] = Form(None)):
    img = await image.read()
    sid = str(uuid.uuid4())[:12]
    app_state.sessions[sid] = {"created": time.time(), "image": img, "question": question, "turns": []}
    if app_state.medgemma_model:
        from PIL import Image as PILImage
        import io
        pil = PILImage.open(io.BytesIO(img)).convert("RGB")
        interp = app_state.medgemma_model.interpret(pil, question or "Describe findings in this medical image.")
    else:
        interp = _demo_interp(question)
    return {"session_id": sid, "status": "session_started", "interpretation": interp, "clarifying_questions": ["Any specific symptoms?", "Previous imaging available?", "Suspected diagnosis?"], "requires_review": True}


@app.post("/api/interpret/continue/{sid}")
async def continue_session(sid: str, answer: Optional[str] = Form(None), audio: Optional[UploadFile] = File(None)):
    if sid not in app_state.sessions:
        raise HTTPException(404, "Session not found")
    s = app_state.sessions[sid]
    if audio and not answer:
        ab = await audio.read()
        answer = app_state.medasr_model.transcribe(ab) if app_state.medasr_model else "(voice - demo)"
    s["turns"].append({"input": answer, "ts": time.time()})
    refined = app_state.medgemma_model.refine(s["image"], answer, s["turns"]) if app_state.medgemma_model else _demo_refine(answer, len(s["turns"]))
    return {"session_id": sid, "refined_interpretation": refined, "turn": len(s["turns"])}


@app.post("/api/interpret/finalize/{sid}")
async def finalize(sid: str):
    if sid not in app_state.sessions:
        raise HTTPException(404, "Session not found")
    s = app_state.sessions.pop(sid)
    return {"session_id": sid, "status": "completed", "final_report": _final_report(sid, s), "total_turns": len(s["turns"]), "requires_clinician_review": True}


@app.post("/api/sync/queue")
async def sync_queue():
    return {"status": "queued"}

@app.get("/api/sync/status")
async def sync_status():
    return {"pending": 0}


def _demo_interp(q):
    return f"RADIOLOGY INTERPRETATION (Demo)\n{'='*50}\nQuestion: {q or 'General'}\n\nFINDINGS:\n- Image adequate for evaluation\n- No acute abnormality identified\n- Heart size normal\n- No pleural effusion\n\nIMPRESSION: No acute findings. Clinical correlation recommended."


def _demo_refine(answer, turn):
    return f"REFINED (Turn {turn})\nContext: {answer}\nAssessment updated with clinical context."


def _final_report(sid, s):
    turns = "\n".join(f"  Turn {i+1}: {t['input']}" for i, t in enumerate(s["turns"]))
    return f"FINAL RADIOLOGY REPORT\nSession: {sid}\nTurns: {len(s['turns'])}\n{turns}\n\nRECOMMENDATIONS:\n1. Clinical correlation essential\n2. Radiologist review required\n\nDISCLAIMER: AI-assisted. Clinician review mandatory."


@app.exception_handler(RequestValidationError)
async def val_err(r, e):
    return JSONResponse(422, {"error": str(e)})

@app.exception_handler(Exception)
async def gen_err(r, e):
    logger.error("Unhandled: %s", e)
    return JSONResponse(500, {"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
