"""OCR using RapidOCR 3.x.

PP-OCRv6 det+rec ONNX models ship inside the rapidocr wheel — nothing is
downloaded at runtime.  One engine is loaded per process; rapidocr 3.x
instances are not documented as thread-safe and this pipeline calls OCR
from background threads, so each inference is serialized under ``_lock``
(the ONNX ops already saturate the runner's cores internally).
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

from PIL import Image
from rapidocr import RapidOCR

_lock = threading.Lock()
_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = RapidOCR()
    return _engine


@dataclass
class OCRBlock:
    text: str
    confidence: float
    box: list


def ocr_image(image_bytes: bytes) -> list[OCRBlock]:
    """Run OCR on raw image bytes. Returns recognized blocks ([] on any
    decode/engine failure — never raises for normal failures)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        import numpy as np
        arr = np.array(img)
    except Exception as e:
        print(f"[OCR] image decode failed: {type(e).__name__}: {e}")
        return []

    engine = _get_engine()
    try:
        with _lock:
            result = engine(arr)
    except Exception as e:
        print(f"[OCR] engine call failed: {type(e).__name__}: {e}")
        return []

    txts = getattr(result, "txts", None)
    if result is None or not txts:
        return []

    scores = list(getattr(result, "scores", None) or [])
    boxes = list(getattr(result, "boxes", None) or [])
    blocks = []
    for i, text in enumerate(txts):
        if not text or not str(text).strip():
            continue
        try:
            conf = float(scores[i]) if i < len(scores) else 1.0
        except (TypeError, ValueError):
            conf = 1.0
        box = (boxes[i].tolist()
               if i < len(boxes) and hasattr(boxes[i], "tolist") else [])
        blocks.append(OCRBlock(text=str(text).strip(), confidence=conf, box=box))
    return blocks


def ocr_image_text(image_bytes: bytes) -> str:
    """Convenience: OCR an image and return all recognized text joined by newlines."""
    blocks = ocr_image(image_bytes)
    return "\n".join(b.text for b in blocks)
