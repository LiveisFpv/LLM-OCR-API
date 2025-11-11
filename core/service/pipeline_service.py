from __future__ import annotations

import time
import os
from typing import Optional

from core.domain.ports.Pipeline_interface import Pipeline_interface
from core.domain.schemas.input_data import InputData
from core.domain.schemas.result_data import MetaInfo, ResultData

from core.lib.logger import get_logger
from .preprocess_service import PreprocessService
from .ocr_service import OCRService
from .layout_service import LayoutService
from .field_extractor_service import FieldExtractorService


class PipelineService(Pipeline_interface):
    def __init__(self) -> None:
        self.logger = get_logger("pipeline")
        self.preprocess = PreprocessService()
        self.ocr = OCRService()
        self.layout = LayoutService()
        self.extractor = FieldExtractorService()

    def run(self, input_data: InputData) -> ResultData:
        t0 = time.perf_counter()
        meta = MetaInfo(request_id=input_data.context.request_id or None, timings_ms={})

        self.logger.info("start pipeline: preprocess_type=%s", input_data.options.preprocess_type.name)

        # 1) Preprocess
        img_data, layout_ir = self.preprocess.run(input_data)
        meta.timings_ms["preprocess"] = int((time.perf_counter() - t0) * 1000)
        if img_data:
            self.logger.info("preprocess: produced %d image page(s)", len(img_data.pages))
        if layout_ir:
            self.logger.info("preprocess: extracted text layout with %d page(s)", len(layout_ir.pages))

        # 2) OCR if needed
        if layout_ir is None and img_data is not None:
            t1 = time.perf_counter()
            ocr = self.ocr.run(img_data)
            meta.timings_ms["ocr"] = int((time.perf_counter() - t1) * 1000)
            total_lines = sum(len(p.lines) for p in ocr.pages)
            self.logger.info("ocr: pages=%d, lines=%d", len(ocr.pages), total_lines)
            os.makedirs("tmp", exist_ok=True)
            for idx, page in enumerate(ocr.pages, start=1):
                dump_path = os.path.join("tmp", f"ocr_lines_{idx}.json")
                try:
                    with open(dump_path, "w", encoding="utf-8") as dump:
                        dump.write(page.model_dump_json())
                    self.logger.info("ocr lines dump saved: %s", dump_path)
                except Exception as exc:
                    self.logger.warning("failed to write %s: %s", dump_path, exc)
            # Sample recognized text
            sample = []
            for p in ocr.pages:
                for ln in p.lines:
                    if ln.text:
                        sample.append(ln.text)
                    if len(sample) >= 10:
                        break
                if len(sample) >= 10:
                    break
            if sample:
                self.logger.info("ocr sample: %s", " | ".join(sample))
            else:
                self.logger.warning("ocr produced 0 text lines; inspect tmp/ocr_lines_*.json and tmp/preprocessed_*.png")
            # Save overlays for debugging
            try:
                os.makedirs("tmp", exist_ok=True)
                import cv2  # local import to avoid hard dep during compile
                for i, p in enumerate(ocr.pages, start=1):
                    canvas = img_data.pages[i - 1].ensure_array().copy()
                    if len(canvas.shape) == 2:
                        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                    for ln in p.lines[:200]:
                        x0, y0, x1, y1 = ln.bbox
                        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    cv2.imwrite(f"tmp/ocr_overlay_{i}.png", canvas)
                self.logger.info("ocr overlays saved: tmp/ocr_overlay_*.png")
            except Exception:
                pass

            # 3) Layout from OCR
            t2 = time.perf_counter()
            layout_ir = self.layout.find_blocks_ocr(ocr)
            meta.timings_ms["layout"] = int((time.perf_counter() - t2) * 1000)
            total_blocks = sum(len(p.blocks) for p in layout_ir.pages)
            self.logger.info("layout: pages=%d, blocks=%d", len(layout_ir.pages), total_blocks)
            # Layout sample
            sample_b = []
            for pg in layout_ir.pages:
                for b in pg.blocks:
                    if b.text:
                        sample_b.append(b.text[:80])
                    if len(sample_b) >= 10:
                        break
                if len(sample_b) >= 10:
                    break
            if sample_b:
                self.logger.info("layout sample: %s", " | ".join(sample_b))

        # 4) Field extraction
        t3 = time.perf_counter()
        result = self.extractor.extract(layout_ir)  # type: ignore[arg-type]
        meta.timings_ms["extract"] = int((time.perf_counter() - t3) * 1000)
        self.logger.info(
            "extracted: org=%s; patient=%s; direction=%s",
            result.source_org.name,
            result.patient.full_name,
            result.direction_type,
        )

        total_ms = int((time.perf_counter() - t0) * 1000)
        self.logger.info("done: total=%d ms", total_ms)
        meta.timings_ms["total"] = total_ms

        return ResultData(meta=meta, layout_ir=layout_ir, result=result)
