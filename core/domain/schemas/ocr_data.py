from typing import List, Optional

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    text: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    conf: float = Field(..., ge=0.0, le=1.0)


class OCRPage(BaseModel):
    num: int
    width: Optional[int] = None
    height: Optional[int] = None
    rotation: int = 0
    lines: List[OCRLine] = Field(default_factory=list)


class OCRData(BaseModel):
    language: Optional[str] = None
    has_text_layer: Optional[bool] = None
    pages: List[OCRPage] = Field(default_factory=list)
