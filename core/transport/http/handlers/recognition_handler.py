from __future__ import annotations

import mimetypes
import os
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from pydantic import AnyHttpUrl

from core.domain.schemas.input_data import DocumentPayload, InputData, ProcessingOptions
from core.domain.schemas.preprocess_type import Preprocess_type
from core.service.pipeline_service import PipelineService
from core.domain.schemas.result_data import ResultData


router = APIRouter()


def _guess_mime(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    mime, _ = mimetypes.guess_type(filename)
    return mime


@router.post("/recognize", summary="Run OCR pipeline for an uploaded file or URL", response_model=ResultData)
async def recognize(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="PDF or image to process"),
    url: Optional[AnyHttpUrl] = Query(default=None, description="Public URL of the document"),
    text_first: bool = Query(default=False, description="Prefer text layer (PDF) before OCR"),
    max_pages: Optional[int] = Query(default=None, ge=1, description="Max pages to process for PDFs"),
) -> ResultData:
    if not file and not url:
        raise HTTPException(status_code=400, detail="either file or url must be provided")

    # Build input payload
    if file is not None:
        content = await file.read()
        payload = DocumentPayload(
            data=content,
            filename=file.filename,
            content_type=file.content_type or _guess_mime(file.filename),
            size_bytes=len(content),
        )
    else:
        payload = DocumentPayload(
            url=str(url),
            filename=os.path.basename(str(url)) or None,
        )

    options = ProcessingOptions(
        preprocess_type=Preprocess_type.TEXT if text_first else Preprocess_type.IMAGE,
        max_pages=max_pages,
    )
    input_data = InputData(document=payload, options=options)

    # Execute pipeline
    # Use pre-initialized pipeline from app state when available
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        pipeline = PipelineService()
    try:
        result = pipeline.run(input_data)
    except HTTPException:
        raise
    except Exception as e:
        # Keep message short for client; details are in server logs
        raise HTTPException(status_code=500, detail=f"processing failed: {e}") from e

    # FastAPI will serialize pydantic models
    return result
