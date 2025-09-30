from __future__ import annotations

from typing import Optional

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData
from core.domain.ports.OCR_provider import OCR_provider
from core.lib.logger import get_logger
from core.local_ai_models.ocr import RapidOCRProvider


class OCRService:
    """High-level OCR service using RapidOCR."""

    def __init__(self, provider: Optional[OCR_provider] = None) -> None:
        logger = get_logger("ocr")
        self.provider = provider or RapidOCRProvider()
        logger.info("OCR provider: RapidOCR")

    def run(self, images: ImageData) -> OCRData:
        return self.provider.get_text(images)
