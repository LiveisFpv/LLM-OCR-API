from abc import ABC, abstractmethod

from core.domain.schemas.result_data import Result


class LLM_provider(ABC):
    @abstractmethod
    def normalize(self, data: Result) -> Result:
        """Optionally normalize and correct extracted fields via LLM."""
        pass
