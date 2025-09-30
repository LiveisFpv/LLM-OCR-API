from __future__ import annotations

import os
from typing import Optional

import numpy as np

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData, OCRLine, OCRPage
from core.domain.ports.OCR_provider import OCR_provider

from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion, LangRec, LangDet

def _quad_to_bbox(quad: list[list[float]]) -> list[int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _any_to_bbox(geom, img_shape) -> list[int]:
    try:
        if isinstance(geom, (list, tuple)):
            if geom and isinstance(geom[0], (list, tuple)):
                return _quad_to_bbox(geom)
            if len(geom) >= 4 and all(isinstance(v, (int, float)) for v in geom[:4]):
                x0, y0, a, b = geom[:4]
                x1, y1 = int(a), int(b)
                x0, y0 = int(x0), int(y0)
                if x1 <= x0 or y1 <= y0:
                    x1 = x0 + int(a)
                    y1 = y0 + int(b)
                return [x0, y0, x1, y1]
    except Exception:
        pass
    h, w = img_shape[:2]
    return [0, 0, int(w), int(h)]


def _enum_value(enum_cls, value: Optional[str], default):
    if not value:
        return default
    try:
        return getattr(enum_cls, value.upper())
    except AttributeError:
        return default


class RapidOCRProvider(OCR_provider):
    """RapidOCR wrapper that auto-downloads Russian models."""

    def __init__(self,
                 *,
                 min_conf: float = 0.0,
                 params: Optional[dict[str, str]] = None) -> None:
        rec_engine = _enum_value(EngineType, (params or {}).get('Rec.engine_type') or os.environ.get('RAPID_REC_ENGINE'), EngineType.PADDLE)
        rec_model_type = _enum_value(ModelType, (params or {}).get('Rec.model_type') or os.environ.get('RAPID_REC_MODEL_TYPE'), ModelType.SERVER)
        rec_version = _enum_value(OCRVersion, (params or {}).get('Rec.ocr_version') or os.environ.get('RAPID_REC_VERSION'), OCRVersion.PPOCRV5)
        rec_lang = _enum_value(LangRec, (params or {}).get('Rec.lang_type') or os.environ.get('RAPID_REC_LANG'), LangRec.ESLAV)

        det_engine = _enum_value(EngineType, (params or {}).get('Det.engine_type') or os.environ.get('RAPID_DET_ENGINE'), None)
        det_model_type = _enum_value(ModelType, (params or {}).get('Det.model_type') or os.environ.get('RAPID_DET_MODEL_TYPE'), None)
        det_version = _enum_value(OCRVersion, (params or {}).get('Det.ocr_version') or os.environ.get('RAPID_DET_VERSION'), None)
        det_lang = _enum_value(LangDet, (params or {}).get('Det.lang_type') or os.environ.get('RAPID_DET_LANG'), None)

        rapid_params: dict[str, object] = {
            "Rec.engine_type": rec_engine,
            "Rec.model_type": rec_model_type,
            "Rec.ocr_version": rec_version,
            "Rec.lang_type": rec_lang,
        }
        if det_engine:
            rapid_params["Det.engine_type"] = det_engine
        if det_model_type:
            rapid_params["Det.model_type"] = det_model_type
        if det_version:
            rapid_params["Det.ocr_version"] = det_version
        if det_lang:
            rapid_params["Det.lang_type"] = det_lang

        self._engine = RapidOCR(params=rapid_params)
        self._min_conf = min_conf

    def get_text(self, data: ImageData) -> OCRData:
        pages: list[OCRPage] = []
        for idx, page in enumerate(data.pages, start=1):
            mat = page.ensure_array()
            ocr_page = self._extract_page(mat, idx)
            pages.append(ocr_page)
        return OCRData(language="ru", has_text_layer=False, pages=pages)

    def _extract_page(self, image: "np.ndarray", page_number: int) -> OCRPage:
        result = self._engine(image)  # RapidOCROutput by default

        lines: list[OCRLine] = []

        if hasattr(result, "boxes") and hasattr(result, "txts") and hasattr(result, "scores"):
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                try:
                    conf = float(score)
                    if conf < self._min_conf:
                        continue
                    # box: np.ndarray shape=(4,2) -> list[[x,y],...]
                    quad = box.tolist() if hasattr(box, "tolist") else box
                    bbox = _any_to_bbox(quad, image.shape)
                    lines.append(OCRLine(text=str(text).strip(), bbox=bbox, conf=conf))
                except Exception:
                    continue

        else:
            out = result
            # tuple: ([...triplets...], elapsed)
            if isinstance(out, (list, tuple)) and len(out) == 2 and isinstance(out[0], (list, tuple)):
                out = out[0]
            if isinstance(out, (list, tuple)):
                for det in out:
                    try:
                        geom = det[0] if len(det) > 0 else None
                        raw_text = str(det[1]) if len(det) > 1 else ""
                        conf = float(det[2]) if len(det) > 2 else 0.0
                    except Exception:
                        continue
                    if conf < self._min_conf:
                        continue
                    bbox = _any_to_bbox(geom, image.shape)
                    lines.append(OCRLine(text=raw_text.strip(), bbox=bbox, conf=conf))

        h, w = image.shape[:2]
        return OCRPage(num=page_number, width=w, height=h, rotation=0, lines=lines)



