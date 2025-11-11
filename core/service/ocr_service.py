from __future__ import annotations
import os
from typing import Optional

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData
from core.domain.ports.OCR_provider import OCR_provider
from core.lib.logger import get_logger

# Оставляем RapidOCR как fallback
from core.local_ai_models.ocr import RapidOCRProvider  # твой текущий провайдер
# Подключаем новый
from core.local_ai_models.deepseek_ocr import DeepSeekOCRProvider


class OCRService:
    """High-level OCR service with switchable provider."""

    def __init__(self, provider: Optional[OCR_provider] = None) -> None:
        logger = get_logger("ocr")
        if provider is not None:
            self.provider = provider
        else:
            which = os.getenv("OCR_PROVIDER", "rapidocr").lower()
            if which == "deepseek":
                self.provider = DeepSeekOCRProvider(
                    model_name=os.getenv("DS_MODEL", "deepseek-ai/DeepSeek-OCR"),
                    backend=os.getenv("DS_BACKEND", "hf"),
                    prompt=os.getenv("DS_PROMPT", "<image>\nFree OCR."),
                )
                logger.info("OCR provider: DeepSeek-OCR (%s backend)", os.getenv("DS_BACKEND", "hf"))
            else:
                self.provider = RapidOCRProvider()
                logger.info("OCR provider: RapidOCR")

    def run(self, images: ImageData) -> OCRData:
        return self.provider.get_text(images)
