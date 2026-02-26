"""
VoiceRad API -- Comprehensive Test Suite
Run:  DEMO_MODE=1 pytest tests/test_api.py -v
"""

import io
import pytest
from fastapi.testclient import TestClient

# Force demo mode so tests never need GPU/models
import os

os.environ["DEMO_MODE"] = "1"

from backend.app import app, app_state  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Ensure a clean session store for every test."""
    app_state.sessions.clear()
    yield
    app_state.sessions.clear()


client = TestClient(app)


# --- Helpers ------------------------------------------------
def _create_test_png():
    """Create a valid minimal PNG image in memory."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()

TINY_PNG = _create_test_png()

FAKE_AUDIO = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 32


def _make_image_file(data: bytes = TINY_PNG, name: str = "xray.png"):
    return ("image", (name, io.BytesIO(data), "image/png"))


def _make_audio_file(data: bytes = FAKE_AUDIO, name: str = "rec.wav"):
    return ("audio", (name, io.BytesIO(data), "audio/wav"))


def _start_session(question: str = "Check this X-ray") -> dict:
    resp = client.post(
        "/api/interpret/start-session",
        files=[_make_image_file()],
        data={"question": question},
    )
    assert resp.status_code == 200
    return resp.json()


# --- Root & Health ------------------------------------------
class TestRootAndHealth:
    def test_root_returns_api_info(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "VoiceRad API"
        assert body["version"] == "1.3.0"
        assert "docs" in body

    def test_health_returns_healthy(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "device" in body
        assert "demo_mode" in body
        assert isinstance(body["models"], dict)
        assert isinstance(body["active_sessions"], int)


# --- Voice Transcription ------------------------------------
class TestVoiceTranscribe:
    def test_transcribe_returns_transcript(self):
        resp = client.post(
            "/api/voice/transcribe",
            files=[_make_audio_file()],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "transcript" in body
        assert isinstance(body["transcript"], str)
        assert len(body["transcript"]) > 0
        assert "confidence" in body

    def test_transcribe_empty_audio_returns_422(self):
        resp = client.post(
            "/api/voice/transcribe",
            files=[("audio", ("empty.wav", io.BytesIO(b""), "audio/wav"))],
        )
        assert resp.status_code == 422


# --- Image Upload -------------------------------------------
class TestImageUpload:
    def test_upload_image_returns_metadata(self):
        resp = client.post(
            "/api/images/upload",
            files=[_make_image_file()],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["image_id"].startswith("img_")
        assert body["filename"] == "xray.png"
        assert body["size_bytes"] > 0
        assert "is_dicom" in body

    def test_upload_empty_image_returns_422(self):
        resp = client.post(
            "/api/images/upload",
            files=[("image", ("empty.png", io.BytesIO(b""), "image/png"))],
        )
        assert resp.status_code == 422

    def test_upload_oversized_image_returns_413(self):
        big = b"x" * (50_000_001)
        resp = client.post(
            "/api/images/upload",
            files=[("image", ("big.png", io.BytesIO(big), "image/png"))],
        )
        assert resp.status_code == 413


# --- Interpretation Session ---------------------------------
class TestInterpretSession:
    def test_start_session_creates_session(self):
        body = _start_session()
        assert body["status"] == "session_started"
        assert "session_id" in body
        assert "interpretation" in body
        assert "RADIOLOGY INTERPRETATION" in body["interpretation"]
        assert isinstance(body["clarifying_questions"], list)

    def test_start_session_empty_image_returns_422(self):
        resp = client.post(
            "/api/interpret/start-session",
            files=[("image", ("e.png", io.BytesIO(b""), "image/png"))],
        )
        assert resp.status_code == 422

    def test_continue_session_with_text(self):
        session = _start_session()
        sid = session["session_id"]
        resp = client.post(
            f"/api/interpret/continue/{sid}",
            data={"answer": "Patient has cough for 2 weeks"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["turn"] == 1
        assert "refined_interpretation" in body

    def test_continue_session_multiple_turns(self):
        session = _start_session()
        sid = session["session_id"]
        for i in range(3):
            resp = client.post(
                f"/api/interpret/continue/{sid}",
                data={"answer": f"Additional context turn {i + 1}"},
            )
            assert resp.status_code == 200
            assert resp.json()["turn"] == i + 1

    def test_continue_nonexistent_session_returns_404(self):
        resp = client.post(
            "/api/interpret/continue/does-not-exist",
            data={"answer": "hello"},
        )
        assert resp.status_code == 404

    def test_continue_without_answer_or_audio_returns_422(self):
        session = _start_session()
        sid = session["session_id"]
        resp = client.post(f"/api/interpret/continue/{sid}")
        assert resp.status_code == 422

    def test_finalize_returns_report(self):
        session = _start_session()
        sid = session["session_id"]
        client.post(
            f"/api/interpret/continue/{sid}",
            data={"answer": "fever and cough"},
        )
        resp = client.post(f"/api/interpret/finalize/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert "FINAL RADIOLOGY REPORT" in body["final_report"]
        assert body["total_turns"] == 1
        assert body["requires_clinician_review"] is True

    def test_finalize_removes_session(self):
        session = _start_session()
        sid = session["session_id"]
        resp1 = client.post(f"/api/interpret/finalize/{sid}")
        assert resp1.status_code == 200
        resp2 = client.post(f"/api/interpret/finalize/{sid}")
        assert resp2.status_code == 404

    def test_finalize_nonexistent_session_returns_404(self):
        resp = client.post("/api/interpret/finalize/nope")
        assert resp.status_code == 404


# --- TTS Endpoint -------------------------------------------
class TestTTS:
    def test_tts_returns_text(self):
        resp = client.post(
            "/api/tts/speak",
            data={"text": "No acute findings observed."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "use_client_tts"
        assert "No acute findings" in body["text"]

    def test_tts_empty_text_returns_422(self):
        resp = client.post("/api/tts/speak", data={"text": ""})
        assert resp.status_code == 422


# --- DICOM Conversion ---------------------------------------
class TestDICOMConversion:
    def test_convert_non_dicom_returns_400(self):
        resp = client.post(
            "/api/images/convert-dicom",
            files=[_make_image_file()],
        )
        assert resp.status_code == 400

    def test_convert_empty_file_returns_422(self):
        resp = client.post(
            "/api/images/convert-dicom",
            files=[("image", ("e.dcm", io.BytesIO(b""), "application/dicom"))],
        )
        assert resp.status_code == 422


# --- Sync Endpoints -----------------------------------------
class TestSync:
    def test_sync_queue(self):
        resp = client.post("/api/sync/queue")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_sync_status(self):
        resp = client.get("/api/sync/status")
        assert resp.status_code == 200
        assert "pending" in resp.json()


# --- Session Management -------------------------------------
class TestSessionManagement:
    def test_session_ttl_expiry(self):
        import time as _time
        session = _start_session()
        sid = session["session_id"]
        app_state.sessions[sid]["created"] = _time.time() - 2000
        resp = client.post(f"/api/interpret/finalize/{sid}")
        assert resp.status_code == 404

    def test_max_sessions_cap(self):
        from backend.app import MAX_SESSIONS
        for _ in range(MAX_SESSIONS):
            _start_session()
        resp = client.post(
            "/api/interpret/start-session",
            files=[_make_image_file()],
            data={"question": "overflow"},
        )
        assert resp.status_code == 429

    def test_health_reports_active_sessions(self):
        _start_session()
        _start_session()
        resp = client.get("/api/health")
        assert resp.json()["active_sessions"] == 2
