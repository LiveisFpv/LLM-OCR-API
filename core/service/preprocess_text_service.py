from __future__ import annotations

from typing import List, Optional

from core.domain.schemas.input_data import InputData
from core.domain.schemas.layout import (
    LayoutBlock,
    LayoutMetadata,
    LayoutPage,
    LayoutRelation,
    LayoutsData,
)
from core.domain.ports.Preprocess_provider import Preprocess_text


class PreprocessTextService(Preprocess_text):
    """Extract a rough text layout from PDF using PyMuPDF text blocks.

    Intended for text-first PDFs; for image-first docs prefer CV + OCR.
    """

    def get_layout(self, input: InputData) -> LayoutsData:
        return self.process_pdf(input)

    def process_pdf(self, input: InputData) -> LayoutsData:
        data, filename, content_type = self._load_bytes(input)
        pages: List[LayoutPage] = []

        try:
            import fitz  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pymupdf is required to parse pdf text layer") from e

        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                blocks: List[LayoutBlock] = []
                reading_order: List[str] = []
                for i, b in enumerate(page.get_text("blocks")):
                    x0, y0, x1, y1, text, *_ = (*b,)
                    bid = f"b{i+1}"
                    blk = LayoutBlock(
                        id=bid,
                        type="text",
                        bbox=[int(x0), int(y0), int(x1), int(y1)],
                        text=text.strip(),
                        source="pdf_text",
                    )
                    blocks.append(blk)
                    reading_order.append(bid)
                p = LayoutPage(
                    num=page.number + 1,
                    width=int(page.rect.width),
                    height=int(page.rect.height),
                    rotation=int(page.rotation or 0),
                    blocks=blocks,
                    reading_order=reading_order,
                )
                pages.append(p)

        md = LayoutMetadata(source="pdf_text", has_text_layer=True)
        return LayoutsData(doc_id=filename or "document", version="1.0", metadata=md, pages=pages)

    def process_document(self, input: InputData) -> LayoutsData:
        return self.process_pdf(input)

    # ------------- helpers -------------
    def _load_bytes(self, input: InputData) -> tuple[bytes, Optional[str], Optional[str]]:
        if input.document.data is not None:
            return input.document.data, input.document.filename, input.document.content_type
        if input.document.url:
            try:
                import requests
                resp = requests.get(str(input.document.url), timeout=15)
                resp.raise_for_status()
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"failed to download document: {e}")
            ct = resp.headers.get("Content-Type")
            return resp.content, input.document.filename, ct
        raise ValueError("no document data provided")

