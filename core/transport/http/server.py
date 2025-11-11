from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .router import api_router
from core.service.pipeline_service import PipelineService


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


def create_app() -> FastAPI:
    load_dotenv()

    swagger_enabled = _bool_env("SWAGGER_ENABLED", True)
    docs_url = "/docs" if swagger_enabled else None
    redoc_url = "/redoc" if swagger_enabled else None

    app = FastAPI(
        title="OCR API",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url="/openapi.json" if swagger_enabled else None,
    )

    # CORS
    raw_origins = os.getenv("ALLOWED_CORS_ORIGINS", "*")
    origins: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(api_router)

    @app.get("/health", tags=["health"])  # simple health endpoint
    def health() -> dict:
        return {"status": "ok"}

    # Warm-up: initialize pipeline and OCR provider once at startup
    # so models are not reloaded per request.
    app.state.pipeline = PipelineService()

    return app


app = create_app()
