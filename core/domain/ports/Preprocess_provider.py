from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.domain.schemas.image_data import ImageData
from core.domain.schemas.input_data import InputData
from core.domain.schemas.layout import LayoutsData

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

class Preprocess_init(ABC):
    @abstractmethod
    def select_preprocess(self, input: InputData) -> InputData:
        pass

class Preprocess_text(ABC):
    @abstractmethod
    def get_layout(self, input: InputData) -> LayoutsData:
        pass

    @abstractmethod
    def process_pdf(self, input: InputData) -> LayoutsData:
        pass

    @abstractmethod
    def process_document(self, input: InputData) -> LayoutsData:
        pass


class Preprocess_cv(ABC):
    @abstractmethod
    def get_image(self, input: InputData) -> ImageData:
        pass

    @abstractmethod
    def _deskew_image(self, image: "np.ndarray") -> "np.ndarray":
        pass

    @abstractmethod
    def _denoise_image(self, image: "np.ndarray") -> "np.ndarray":
        pass

    @abstractmethod
    def _binarize_image(self, image: "np.ndarray") -> "np.ndarray":
        pass

    @abstractmethod
    def _correct_perspective_image(self, image: "np.ndarray") -> "np.ndarray":
        pass

    
