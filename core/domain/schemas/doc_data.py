from typing import List, Optional

from pydantic import BaseModel, Field


class DocLine(BaseModel):
    text: str
    bbox: Optional[List[int]] = Field(default=None, min_items=4, max_items=4)


class DocPage(BaseModel):
    num: int
    width: Optional[int] = None
    height: Optional[int] = None
    rotation: int = 0
    text_lines: List[DocLine] = Field(default_factory=list)


class DocData(BaseModel):
    language: Optional[str] = None
    pages: List[DocPage] = Field(default_factory=list)

