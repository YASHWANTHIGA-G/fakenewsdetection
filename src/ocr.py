"""
Verdict - Optical Character Recognition (OCR) Engine.

Extracts plain text from images (screenshots, WhatsApp forwards, news clippings, memes)
using EasyOCR, with Pillow-based image enhancement and lazy model loading.
Also notes pytesseract as a system-dependent alternative.
"""

from __future__ import annotations

import io
from typing import Any, Optional

from PIL import Image, ImageEnhance, ImageOps

# Singleton instance for lazy loading EasyOCR
_EASYOCR_READER: Optional[Any] = None


def get_ocr_reader() -> Any:
    """
    Lazy-load and cache the EasyOCR Reader instance (English by default).
    Downloading/loading weights happens only once across requests.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import os
            import sys
            os.environ["PYTHONIOENCODING"] = "utf-8"
            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            import easyocr

            # gpu=False for CPU portability; verbose=False prevents Windows console encoding crash on download progress bar
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Run 'pip install easyocr' or use "
                "pytesseract with a local Tesseract-OCR binary installation."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR engine: {e}")
    return _EASYOCR_READER


def preprocess_image_for_ocr(image_bytes: bytes) -> Image.Image:
    """
    Preprocess image bytes to maximize OCR accuracy:
    1. Open and convert to RGB (strips alpha/transparency channels).
    2. Resize proportionally if width is under 1000px to ensure legible letter heights.
    3. Convert to grayscale.
    4. Enhance contrast to separate ink from paper/background noise.
    """
    stream = io.BytesIO(image_bytes)
    image = Image.open(stream)

    # 1. Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Upscale small images if width is under 1000px
    orig_w, orig_h = image.size
    min_width = 1000
    if orig_w < min_width and orig_w > 0:
        scale_ratio = min_width / orig_w
        new_w = min_width
        new_h = int(orig_h * scale_ratio)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3. Convert to grayscale for OCR thresholding
    gray_image = image.convert("L")

    # 4. Increase contrast using Pillow ImageEnhance
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(1.8)

    # Optional auto-contrast to stretch dynamic range
    enhanced_image = ImageOps.autocontrast(enhanced_image)

    return enhanced_image


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract readable text from an image byte stream.

    Parameters:
    ----------
    file_bytes : bytes
        Raw binary content of the uploaded image (PNG, JPG, JPEG, WEBP).

    Returns:
    -------
    str : Extracted text joined with standard spacing.

    Raises:
    ------
    ValueError : If image is invalid or fewer than 15 words are found.
    """
    if not file_bytes:
        raise ValueError("Empty image file provided.")

    try:
        processed_img = preprocess_image_for_ocr(file_bytes)
    except Exception as e:
        raise ValueError(f"Unable to read or parse image file: {e}")

    # Convert PIL image to in-memory bytes for EasyOCR
    img_byte_arr = io.BytesIO()
    processed_img.save(img_byte_arr, format="PNG")
    clean_bytes = img_byte_arr.getvalue()

    reader = get_ocr_reader()

    # EasyOCR returns list of tuples: (bounding_box, text, confidence)
    results = reader.readtext(clean_bytes, detail=0, paragraph=True)

    extracted_text = "\n\n".join(results).strip()
    word_count = len(extracted_text.split())

    if word_count < 15:
        raise ValueError(
            "We couldn't read enough text from this image. "
            "Try a sharper screenshot or paste the text instead."
        )

    return extracted_text


if __name__ == "__main__":
    print("OCR module loaded. EasyOCR initialized lazily upon first invocation.")
    print("Pytesseract can also be used as a backend alternative if configured.")
