from __future__ import annotations

from typing import List, Tuple

from core.domain.schemas.doc_data import DocData
from core.domain.schemas.ocr_data import OCRData
from core.domain.schemas.layout import LayoutBlock, LayoutMetadata, LayoutPage, LayoutsData
from core.domain.ports.Layout_analyzer import Layout_analyzer


class LayoutService(Layout_analyzer):
    """Very simple layout builder.

    - For OCR input: maps OCR lines to `LayoutBlock(type='text')` blocks
      preserving bbox and confidence.
    - For DocData (text layer): similarly maps to blocks.

    Downstream extractor can group these text blocks into KV pairs/tables.
    """

    def find_blocks_ocr(self, data: OCRData) -> LayoutsData:
        pages: List[LayoutPage] = []
        for page in data.pages:
            blocks: List[LayoutBlock] = []
            order: List[str] = []
            for i, line in enumerate(page.lines, start=1):
                bid = f"b{i}"
                text = (line.text or "").strip()
                btype, label, value = self._classify_line(text)
                blk = LayoutBlock(
                    id=bid,
                    type=btype,
                    bbox=line.bbox,
                    text=value if value is not None else text,
                    conf=line.conf,
                    source="ocr",
                    label=label,
                )
                blocks.append(blk)
                order.append(bid)
            pages.append(
                LayoutPage(
                    num=page.num,
                    width=page.width or 0,
                    height=page.height or 0,
                    rotation=page.rotation,
                    blocks=blocks,
                    reading_order=order,
                )
            )
        md = LayoutMetadata(source="ocr", language=data.language, has_text_layer=False)
        return LayoutsData(doc_id=None, version="1.0", metadata=md, pages=pages)

    def find_blocks_text(self, data: DocData) -> LayoutsData:
        pages: List[LayoutPage] = []
        for page in data.pages:
            blocks: List[LayoutBlock] = []
            order: List[str] = []
            for i, line in enumerate(page.text_lines, start=1):
                bid = f"t{i}"
                text = (line.text or "").strip()
                btype, label, value = self._classify_line(text)
                blk = LayoutBlock(
                    id=bid,
                    type=btype,
                    bbox=line.bbox or [0, 0, 0, 0],
                    text=value if value is not None else text,
                    source="pdf_text",
                    label=label,
                )
                blocks.append(blk)
                order.append(bid)
            pages.append(
                LayoutPage(
                    num=page.num,
                    width=page.width or 0,
                    height=page.height or 0,
                    rotation=page.rotation,
                    blocks=blocks,
                    reading_order=order,
                )
            )
        md = LayoutMetadata(source="pdf_text", language=data.language, has_text_layer=True)
        return LayoutsData(doc_id=None, version="1.0", metadata=md, pages=pages)

    def _classify_line(self, text: str) -> Tuple[str, str | None, str | None]:
        """Classify a single text line into a block type.

        Returns (type, label, value):
        - meta_kv for "key: value"
        - meta_kv also for "key - value" and dash variants (–, —)
        - hazard_item for indices like 8.x/9.x (and any N.x generally)
        - heading for likely uppercase headings
        - text otherwise
        """
        t = text.strip()
        if not t:
            return "text", None, t
        # meta key-value by common separators (colon or dashes)
        import re
        # Allow colon or various dash characters as separators
        m_kv = re.match(r"^\s*(.+?)\s*[:\-–—]\s*(.+?)\s*$", t)
        if m_kv:
            key = m_kv.group(1).strip()
            val = m_kv.group(2).strip()
            # Avoid treating hazard headings without values as kv
            if key and val and not re.match(r"^\d+\.\d+\.?$", key):
                return "meta_kv", key, val
        # hazard items like 8.1, 9.4, etc. (allow trailing dot after index)
        m = re.match(r"^\s*(\d+\.\d+)\.?(.*)$", t)
        if m:
            return "hazard_item", m.group(1), m.group(2).strip()
        letters = [ch for ch in t if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.6 and len(t) >= 10:
            return "heading", None, t
        return "text", None, t

