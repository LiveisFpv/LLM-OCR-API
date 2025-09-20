from abc import ABC, abstractmethod

from domain.schemas.input_data import InputData
from domain.schemas.result_data import ResultData


class Pipeline_interface(ABC):
    @abstractmethod
    def run(self, input_data: InputData) -> ResultData:
        pass
