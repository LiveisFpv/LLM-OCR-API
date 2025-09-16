from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LayoutMetadata(BaseModel):
    source: Optional[str] = None
    language: Optional[str] = None
    has_text_layer: Optional[bool] = None
    dpi: Optional[int] = None
    units: Optional[str] = None


class TextLine(BaseModel):
    text: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    conf: Optional[float] = None


class LayoutBlock(BaseModel):
    id: str
    type: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)
    conf: Optional[float] = None
    source: Optional[str] = None

    # Flexible content fields to cover different block kinds
    text: Optional[str] = None
    text_lines: Optional[List[TextLine]] = None
    label: Optional[str] = None
    label_prefix: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = None
    structure: Optional[Dict[str, Any]] = None


class LayoutPage(BaseModel):
    num: int
    width: int
    height: int
    rotation: int = 0
    blocks: List[LayoutBlock] = Field(default_factory=list)
    reading_order: List[str] = Field(default_factory=list)


class LayoutRelation(BaseModel):
    from_: str = Field(alias="from")
    to: str
    type: str


class LayoutsData(BaseModel):
    doc_id: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[LayoutMetadata] = None
    pages: List[LayoutPage] = Field(default_factory=list)
    relations: Optional[List[LayoutRelation]] = None
