from abc import ABC, abstractmethod

from core.domain.schemas.layout import LayoutsData
from core.domain.schemas.result_data import Result


class Field_extractor_provider(ABC):
    @abstractmethod
    def extract(self, layout: LayoutsData) -> Result:
        """Extract all target fields from the unified layout IR.

        Implementations should use anchors/regex/geometry to fill the Result
        structure (source_org, medical_org, patient, employment, hazards).
        """
        pass
