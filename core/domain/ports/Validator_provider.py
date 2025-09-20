from abc import ABC, abstractmethod

from domain.schemas.result_data import Result, ErrorEntry


class Validator_provider(ABC):
    @abstractmethod
    def validate(self, result: Result) -> list[ErrorEntry]:
        """Return a list of validation errors for the extracted fields."""
        pass
