from __future__ import annotations

from typing import Optional, Tuple

from core.domain.schemas.input_data import InputData
from core.domain.schemas.layout import LayoutsData
from core.domain.schemas.image_data import ImageData
from core.domain.schemas.preprocess_type import Preprocess_type

from .preprocess_cv_service import PreprocessCVService
from .preprocess_text_service import PreprocessTextService


class PreprocessService:
    """Orchestrate preprocessing branch selection.

    For now, the TEXT branch uses PDF text layer (PyMuPDF), and the IMAGE branch
    rasterizes to images and runs CV cleanup.
    """

    def __init__(self) -> None:
        self.cv = PreprocessCVService()
        self.text = PreprocessTextService()

    def run(self, input_data: InputData) -> Tuple[Optional[ImageData], Optional[LayoutsData]]:
        ptype = input_data.options.preprocess_type

        if ptype == Preprocess_type.IMAGE:
            return self.cv.get_image(input_data), None

        # Default: try text path, fall back to CV if it fails
        try:
            layout = self.text.get_layout(input_data)
            return None, layout
        except Exception:
            # Fall back to CV rasterization if the PDF has no text layer
            return self.cv.get_image(input_data), None

