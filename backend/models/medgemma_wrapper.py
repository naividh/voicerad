"""
MedGemma 1.5 4B Model Wrapper
Handles medical image + text interpretation for VoiceRad

Fixes applied (v1.2.0):
- 4-bit NF4 quantization (VRAM: ~8GB -> ~3GB, speed: ~3x faster)
- Greedy decoding for deterministic clinical output
- GPU memory cleanup after each inference
- refine() now accepts PIL Image directly (fixes DICOM crash on turn 2+)
- Proper torch.inference_mode() context management
"""

import torch
import logging
import gc
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)


class MedGemmaModel:
    """Wrapper for MedGemma 1.5 4B multimodal model with optimized inference."""

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_4bit = use_4bit and device == "cuda"
        self.model = None
        self.processor = None
        self._load_model()

    # -- Model Loading ----------------------------------------
    def _load_model(self):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.info(
                "Loading %s (4-bit=%s) on %s ...",
                self.model_name, self.use_4bit, self.device,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            load_kwargs = {"low_cpu_mem_usage": True}

            if self.use_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = (
                    torch.float16 if self.device == "cuda" else torch.float32
                )
                load_kwargs["device_map"] = self.device

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model.eval()

            if self.device == "cuda" and torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    "MedGemma loaded | VRAM: %.1f GB | 4-bit: %s",
                    mem_gb, self.use_4bit,
                )
            else:
                logger.info("MedGemma loaded on %s", self.device)

        except Exception as exc:
            logger.error("Failed to load MedGemma: %s", exc)
            raise

    # -- GPU Memory Cleanup -----------------------------------
    def _cleanup_gpu(self):
        """Free fragmented GPU memory after inference."""
        if self.device == "cuda" and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    # -- Core Interpretation ----------------------------------
    def interpret(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Interpret a medical image given a clinical question.
        Uses greedy decoding for deterministic, faster clinical output.
        """
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
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding = faster + deterministic
                )

            new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
            return self.processor.decode(new_tokens, skip_special_tokens=True)
        finally:
            # Always clean up GPU memory
            del inputs
            self._cleanup_gpu()

    # -- Contextual Refinement --------------------------------
    def refine(
        self,
        image: Image.Image,
        answer: str,
        turns: list,
        max_tokens: int = 512,
    ) -> str:
        """
        Refine interpretation with additional clinical context.

        CHANGED: Now accepts PIL Image directly instead of raw bytes.
        This fixes the DICOM crash on turn 2+ where raw DICOM bytes
        could not be re-opened with Image.open().
        """
        history = "\n".join(
            f"Turn {i + 1}: {t['input']}" for i, t in enumerate(turns)
        )
        prompt = (
            f"Previous conversation:\n{history}\n\n"
            f"New information: {answer}\n\n"
            "Provide a refined radiology interpretation considering "
            "all clinical context provided."
        )
        return self.interpret(image, prompt, max_tokens)

    # -- Structured Report Generation -------------------------
    def generate_report(
        self,
        image: Image.Image,
        imaging_type: str = "Unknown",
    ) -> dict:
        """Generate a structured radiology report."""
        prompt = (
            f"Analyse this {imaging_type} and provide a structured "
            "radiology report with sections:\n"
            "1. TECHNIQUE\n2. FINDINGS\n3. IMPRESSION\n4. RECOMMENDATIONS"
        )
        report = self.interpret(image, prompt, max_tokens=1024)
        return {
            "imaging_type": imaging_type,
            "report": report,
            "requires_review": True,
        }
