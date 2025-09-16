from pydantic import BaseModel
from .layout import LayoutsData
from typing import Any, Dict, List, Optional

class Result(BaseModel):
    #! Основные поля
    errors: List

class ResultData(BaseModel):
    layout: Optional[LayoutsData]
    result: Result

