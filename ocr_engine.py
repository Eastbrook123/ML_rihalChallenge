"""
OCR engine with EasyOCR primary backend and PaddleOCR fallback.
Returns structured output with bounding boxes, text, and confidence.
"""

from __future__ import annotations

import cv2
import numpy as np


class OCRResult:
    """Holds OCR output for a single image."""

    def __init__(
        self,
        boxes: list,
        texts: list[str],
        confidences: list[float],
        image_shape: tuple,
    ):
        self.boxes = boxes
        self.texts = texts
        self.confidences = confidences
        self.image_shape = image_shape

    def lines_top_to_bottom(self) -> list[tuple]:
        """Return (text, box, confidence) sorted top-to-bottom by box centroid Y."""
        if not self.texts:
            return []
        items = list(zip(self.texts, self.boxes, self.confidences))
        items.sort(key=lambda x: np.mean([pt[1] for pt in x[1]]))
        return items

    @property
    def full_text(self) -> str:
        return "\n".join(t for t, _, _ in self.lines_top_to_bottom())


class OCREngine:
    """Unified OCR engine. Tries EasyOCR first, then PaddleOCR."""

    def __init__(self, backend: str = "auto"):
        self._reader = None
        self._backend = backend

    def _ensure_loaded(self):
        if self._reader is not None:
            return

        if self._backend in ("auto", "easyocr"):
            try:
                import easyocr

                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._backend = "easyocr"
                return
            except ImportError:
                if self._backend == "easyocr":
                    raise

        if self._backend in ("auto", "paddleocr"):
            try:
                from paddleocr import PaddleOCR

                self._reader = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.3,
                    rec_batch_num=8,
                )
                self._backend = "paddleocr"
                return
            except ImportError:
                if self._backend == "paddleocr":
                    raise

        raise ImportError(
            "No OCR backend available. Install easyocr or paddleocr."
        )

    def run(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()

        image = self._preprocess(image)

        h, w = image.shape[:2]
        max_side = 1536
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        if self._backend == "easyocr":
            return self._run_easyocr(image)
        else:
            return self._run_paddleocr(image)

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def _run_easyocr(self, image: np.ndarray) -> OCRResult:
        results = self._reader.readtext(image, width_ths=0.9, paragraph=False)
        boxes, texts, confidences = [], [], []
        for bbox, text, conf in results:
            box = [list(pt) for pt in bbox]
            boxes.append(box)
            texts.append(text)
            confidences.append(conf)
        return OCRResult(boxes, texts, confidences, image.shape)

    def _run_paddleocr(self, image: np.ndarray) -> OCRResult:
        result = self._reader.ocr(image, cls=False)
        boxes, texts, confidences = [], [], []
        if result and result[0]:
            for line in result[0]:
                boxes.append(line[0])
                texts.append(line[1][0])
                confidences.append(line[1][1])
        return OCRResult(boxes, texts, confidences, image.shape)
