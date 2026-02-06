"""
MedASR Model Wrapper  
Converts medical speech to text for VoiceRad
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class MedASRModel:
    """Wrapper for medical speech recognition"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForCTC, AutoProcessor
            model_name = "google/medASR"
            logger.info("Loading MedASR from %s ...", model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("MedASR loaded")
        except Exception:
            logger.warning("MedASR unavailable, trying Whisper fallback")
            try:
                import whisper
                self.model = whisper.load_model("base", device=self.device)
                self.processor = None
                logger.info("Whisper fallback loaded")
            except Exception as exc:
                logger.error("No speech model available: %s", exc)
                raise

    def transcribe(self, audio_data, sample_rate: int = 16000) -> str:
        """Transcribe audio bytes or numpy array to text."""
        import io, torch

        if isinstance(audio_data, bytes):
            import soundfile as sf
            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
        elif isinstance(audio_data, np.ndarray):
            audio_np = audio_data
        else:
            raise ValueError("audio_data must be bytes or numpy array")

        if hasattr(self.model, "transcribe"):  # Whisper
            result = self.model.transcribe(audio_np)
            return result["text"]

        # CTC model path
        inputs = self.processor(
            audio_np, sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0]

    def transcribe_file(self, path: str) -> str:
        """Load an audio file and transcribe it."""
        import librosa
        audio, sr = librosa.load(path, sr=16000)
        return self.transcribe(audio, sample_rate=sr)
