from __future__ import annotations

from fastapi import APIRouter

from .handlers.recognition_handler import router as recognition_router
from .handlers.setting_handler import router as settings_router


api_router = APIRouter()
api_router.include_router(recognition_router, prefix="/api", tags=["recognition"])
api_router.include_router(settings_router, prefix="/api", tags=["settings"])
