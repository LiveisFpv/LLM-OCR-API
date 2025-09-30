from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.ocr_data import OCRData, OCRPage

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class OCR_provider(ABC):
    @abstractmethod
    def get_text(self, data: ImageData) -> OCRData:
        pass

    @abstractmethod
    def _extract_page(self, image: "np.ndarray", page_number: int) -> OCRPage:
        pass
