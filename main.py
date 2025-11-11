from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

from core.domain.schemas.input_data import DocumentPayload, InputData, ProcessingOptions
from core.domain.schemas.preprocess_type import Preprocess_type
from core.service.pipeline_service import PipelineService

from dotenv import load_dotenv
load_dotenv()


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes")


def build_input_from_file(path: Path, *, text_first: bool = False) -> InputData:
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    options = ProcessingOptions(
        preprocess_type=Preprocess_type.TEXT if text_first else Preprocess_type.IMAGE,
        max_pages=int(os.getenv("MAX_PAGES", "3")),
    )
    payload = DocumentPayload(data=data, filename=path.name, content_type=mime)
    return InputData(document=payload, options=options)


def main() -> None:
    serve = _as_bool(os.getenv("SERVE", "0"))
    if serve and len(sys.argv) <= 1:
        # Run HTTP server; host/port from env
        import uvicorn
        host = os.getenv("DOMAIN", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        uvicorn.run("core.transport.http.server:app", host=host, port=port, reload=False)
        return

    # CLI mode: first arg is file path
    if len(sys.argv) <= 1:
        print("Usage: python main.py <file>  # or set SERVE=1 to start HTTP server", flush=True)
        sys.exit(2)

    file_path = sys.argv[1]
    text_first = _as_bool(os.getenv("TEXT_FIRST", "0"))

    input_data = build_input_from_file(Path(file_path), text_first=text_first)
    pipeline = PipelineService()
    _ = pipeline.run(input_data)


if __name__ == "__main__":
    main()
