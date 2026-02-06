"""
MedGemma 1.5 4B Model Wrapper
Handles medical image + text interpretation for VoiceRad
"""

import torch
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)


class MedGemmaModel:
    """Wrapper for MedGemma 1.5 4B multimodal model"""

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────
    def _load_model(self):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.info("Loading %s ...", self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            logger.info("MedGemma loaded on %s", self.device)
        except Exception as exc:
            logger.error("Failed to load MedGemma: %s", exc)
            raise

    # ── Core Interpretation ───────────────────────────────
    def interpret(self, image: Image.Image, question: str, max_tokens: int = 512) -> str:
        """Interpret a medical image given a clinical question."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)

        # Decode only the new tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    # ── Contextual Refinement ─────────────────────────────
    def refine(self, image_bytes: bytes, answer: str, turns: list, max_tokens: int = 512) -> str:
        """Refine interpretation with additional clinical context."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        history = "\n".join(f"Turn {i+1}: {t['input']}" for i, t in enumerate(turns))
        prompt = (
            f"Previous conversation:\n{history}\n\n"
            f"New information: {answer}\n\n"
            "Provide a refined radiology interpretation considering all context."
        )
        return self.interpret(pil, prompt, max_tokens)

    # ── Structured Report Generation ──────────────────────
    def generate_report(self, image: Image.Image, imaging_type: str = "Unknown") -> dict:
        """Generate a structured radiology report."""
        prompt = (
            f"Analyse this {imaging_type} and provide a structured radiology report:\n"
            "1. TECHNIQUE\n2. FINDINGS\n3. IMPRESSION\n4. RECOMMENDATIONS"
        )
        report = self.interpret(image, prompt, max_tokens=1024)
        return {
            "imaging_type": imaging_type,
            "report": report,
            "requires_review": True,
        }
