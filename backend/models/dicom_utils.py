"""
DICOM Image Utilities
Handles DICOM file parsing, metadata extraction, and conversion to PIL Image.
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def is_dicom(data: bytes) -> bool:
      """Check if raw bytes represent a DICOM file."""
      if len(data) < 132:
                return False
            return data[128:132] == b"DICM"


def dicom_to_pil(data: bytes) -> Image.Image:
      """Convert DICOM bytes to a PIL RGB image."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data))
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply windowing if present
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
              center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
              width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
              lower = center - width / 2
              upper = center + width / 2
              pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
              pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    # Handle photometric interpretation (invert if MONOCHROME1)
      if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = 255 - pixel_array

    img = Image.fromarray(pixel_array)
    return img.convert("RGB")


def extract_dicom_metadata(data: bytes) -> dict:
      """Extract clinically relevant metadata from a DICOM file."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data))

    def safe_get(attr: str, default: str = "") -> str:
              val = getattr(ds, attr, default)
              return str(val).strip() if val else default

    return {
              "patient_id": safe_get("PatientID", "Anonymous"),
              "modality": safe_get("Modality", "Unknown"),
              "body_part": safe_get("BodyPartExamined", "Unknown"),
              "study_description": safe_get("StudyDescription"),
              "series_description": safe_get("SeriesDescription"),
              "institution": safe_get("InstitutionName"),
              "study_date": safe_get("StudyDate"),
              "image_dimensions": f"{getattr(ds, 'Rows', '?')}x{getattr(ds, 'Columns', '?')}",
              "bits_stored": safe_get("BitsStored"),
              "photometric_interpretation": safe_get("PhotometricInterpretation"),
    }
