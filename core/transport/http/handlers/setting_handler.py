from __future__ import annotations

import os
from typing import Dict

from fastapi import APIRouter


router = APIRouter()


@router.get("/settings", summary="Current server settings")
def get_settings() -> Dict[str, str]:
    keys = [
        "DOMAIN",
        "PORT",
        "ALLOWED_CORS_ORIGINS",
        "SWAGGER_ENABLED",
        "LLM_API_ADDRESS",
        "MAX_FILE_MB",
        "MAX_PAGES",
        "OCR_PROVIDER",
        "DS_MODEL",
        "DS_BACKEND",
    ]
    return {k: os.getenv(k, "") for k in keys}
