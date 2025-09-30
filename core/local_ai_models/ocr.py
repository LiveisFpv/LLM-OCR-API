from __future__ import annotations

from typing import Optional

import numpy as np

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData, OCRLine, OCRPage
from core.domain.ports.OCR_provider import OCR_provider


def _quad_to_bbox(quad: list[list[float]]) -> list[int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


class PaddleOCRProvider(OCR_provider):
    """Modern local OCR provider based on PaddleOCR (det+cls+rec)."""

    def __init__(self, lang: str = "ru", use_angle_cls: bool = True) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "paddleocr is required. Install: pip install paddleocr paddlepaddle (CPU)"
            ) from e

        self._ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)
        self._lang = lang

    def get_text(self, data: ImageData) -> OCRData:
        pages: list[OCRPage] = []
        for idx, page in enumerate(data.pages, start=1):
            mat = page.ensure_array()
            ocr_page = self._extract_page(mat, idx)
            pages.append(ocr_page)
        return OCRData(language=self._lang, has_text_layer=False, pages=pages)

    def _extract_page(self, image: "np.ndarray", page_number: int) -> OCRPage:  # type: ignore[override]
        img = image
        result = self._ocr.ocr(img, cls=True)

        lines: list[OCRLine] = []
        if result and len(result) > 0:
            for det in result[0]:
                quad, (txt, conf) = det
                bbox = _quad_to_bbox(quad)
                if txt.strip():
                    lines.append(OCRLine(text=txt.strip(), bbox=bbox, conf=float(conf)))

        h, w = img.shape[:2]
        return OCRPage(num=page_number, width=w, height=h, rotation=0, lines=lines)
