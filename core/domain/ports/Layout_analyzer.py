from abc import ABC, abstractmethod

from domain.schemas.ocr_data import OCRData
from domain.schemas.doc_data import DocData
from domain.schemas.layout import LayoutsData


class Layout_analyzer(ABC):
    @abstractmethod
    def find_blocks_ocr(self, data: OCRData) -> LayoutsData:
        pass

    @abstractmethod
    def find_blocks_text(self, data: DocData) -> LayoutsData:
        pass
