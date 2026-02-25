"""
MedASR Model Wrapper
Converts medical speech to text for VoiceRad

Fix applied (v1.2.0):
- Removed fake google/medASR model attempt (model not publicly available)
- Uses OpenAI Whisper directly as primary STT engine
- Added proper error handling and audio format validation
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class MedASRModel:
    """Wrapper for medical speech recognition using Whisper."""

    def __init__(self, device: str = "cuda", model_size: str = "base"):
        self.device = device
        self.model_size = model_size
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model directly (MedASR is not publicly available)."""
        try:
            import whisper

            logger.info("Loading Whisper '%s' on %s ...", self.model_size, self.device)
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.processor = None  # Whisper has its own processor
            logger.info("Whisper '%s' loaded successfully", self.model_size)
        except ImportError:
            logger.error(
                "openai-whisper not installed. "
                "Run: pip install openai-whisper"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            raise

    def transcribe(self, audio_data, sample_rate: int = 16000) -> str:
        """Transcribe audio bytes or numpy array to text."""
        import io
        import torch

        if isinstance(audio_data, bytes):
            if len(audio_data) == 0:
                return ""
            try:
                import soundfile as sf
                audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
            except Exception as exc:
                logger.warning("soundfile failed, trying librosa: %s", exc)
                try:
                    import librosa
                    audio_np, sample_rate = librosa.load(
                        io.BytesIO(audio_data), sr=16000
                    )
                except Exception as exc2:
                    logger.error("All audio decoders failed: %s", exc2)
                    return "(audio format not supported)"
        elif isinstance(audio_data, np.ndarray):
            audio_np = audio_data
        else:
            raise ValueError("audio_data must be bytes or numpy array")

        # Ensure mono audio
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Ensure float32
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        try:
            result = self.model.transcribe(audio_np)
            text = result.get("text", "").strip()
            logger.info("Transcribed %d samples -> '%s'", len(audio_np), text[:50])
            return text
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return "(transcription failed)"

    def transcribe_file(self, path: str) -> str:
        """Load an audio file and transcribe it."""
        import librosa
        audio, sr = librosa.load(path, sr=16000)
        return self.transcribe(audio, sample_rate=sr)
