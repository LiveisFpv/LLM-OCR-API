from abc import ABC, abstractmethod
from typing import Any, Optional


class Settings_provider(ABC):
    @abstractmethod
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        pass
