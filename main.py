from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from core.domain.schemas.input_data import DocumentPayload, InputData, ProcessingOptions
from core.domain.schemas.preprocess_type import Preprocess_type
from core.service.pipeline_service import PipelineService

from dotenv import load_dotenv
load_dotenv()


def build_input_from_file(path: Path, *, text_first: bool = False) -> InputData:
    data = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    options = ProcessingOptions(
        preprocess_type=Preprocess_type.TEXT if text_first else Preprocess_type.IMAGE,
        max_pages=3,
    )
    payload = DocumentPayload(data=data, filename=path.name, content_type=mime)
    return InputData(document=payload, options=options)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OCR pipeline on a local file")
    ap.add_argument("--file", default="photo.jpg", type=str, help="Path to PDF or image")
    ap.add_argument("--text-first", default="",action="store_true", help="Prefer text layer for PDFs")
    args = ap.parse_args()

    input_data = build_input_from_file(Path(args.file), text_first=args.text_first)
    pipeline = PipelineService()
    _ = pipeline.run(input_data)


if __name__ == "__main__":
    main()
