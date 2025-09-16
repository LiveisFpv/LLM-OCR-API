from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImagePage(BaseModel):
    """Single image page that can store either encoded bytes or a decoded matrix."""

    content: Optional[bytes] = None  # encoded image bytes (PNG/JPEG/etc.)
    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None
    mode: Optional[str] = None  # e.g., 'RGB', 'L'
    array: Optional[Any] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def ensure_array(self) -> Any:
        """Return a decoded numpy matrix, decoding the bytes lazily when required."""

        if self.array is not None:
            return self.array
        if self.content is None:
            raise ValueError("image page does not contain encoded bytes to decode")

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("opencv-python and numpy are required to decode image bytes") from exc

        matrix = cv2.imdecode(np.frombuffer(self.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if matrix is None:
            raise ValueError("failed to decode image bytes")

        self.array = matrix
        if matrix is not None:
            self.height, self.width = matrix.shape[:2]
        return matrix

    def set_array(self, matrix: Any, *, encode: bool = False, format: str = ".png") -> None:
        """Store a numpy matrix and optionally refresh the encoded bytes."""

        self.array = matrix
        if matrix is not None:
            self.height, self.width = matrix.shape[:2]
        else:
            self.height = None
            self.width = None

        if not encode:
            return

        if matrix is None:
            raise ValueError("cannot encode an empty image matrix")

        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("opencv-python is required to encode image matrices") from exc

        ok, buffer = cv2.imencode(format, matrix)
        if not ok:
            raise ValueError("failed to encode image matrix")
        self.content = buffer.tobytes()


class ImageData(BaseModel):
    """Collection of preprocessed image pages passed to OCR."""

    pages: List[ImagePage] = Field(default_factory=list)
    source: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def ensure_arrays(self) -> List[Any]:
        """Decode all pages and return their numpy matrices."""

        return [page.ensure_array() for page in self.pages]
