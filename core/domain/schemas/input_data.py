from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .preprocess_type import Preprocess_type


class DocumentPayload(BaseModel):
    """Raw document payload supplied by the client."""

    data: Optional[bytes] = None
    url: Optional[HttpUrl] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    pages: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source(self) -> "DocumentPayload":
        # Require at least one data source so downstream services can fetch the document.
        if self.data is None and self.url is None:
            raise ValueError("document payload requires either inline data or a URL")
        return self


class ProcessingOptions(BaseModel):
    """Flags that control how the pipeline should process the document."""

    preprocess_type: Preprocess_type = Preprocess_type.TEXT
    doc_type: Optional[str] = None
    return_layout_ir: bool = True
    run_llm_normalization: bool = True
    enable_validation: bool = True
    max_pages: Optional[int] = Field(default=None, ge=1)


class RequestContext(BaseModel):
    """Request-scoped metadata propagated through the pipeline."""

    request_id: Optional[str] = None
    timeout_ms: Optional[int] = Field(default=None, ge=0)
    client_tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class InputData(BaseModel):
    """Full payload consumed by the pipeline orchestrator."""

    document: DocumentPayload
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    context: RequestContext = Field(default_factory=RequestContext)
    metadata: Dict[str, Any] = Field(default_factory=dict)
