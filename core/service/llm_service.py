from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from pydantic import BaseModel

from core.domain.schemas.layout import LayoutsData, LayoutBlock
from core.domain.schemas.result_data import Result
from core.lib.logger import get_logger


class LLMService:
    """LLM-based extractor that converts layout IR to Result via Ollama Cloud.

    Controlled by env:
      - OLLAMA_HOST (default: https://ollama.com)
      - OLLAMA_API_KEY (required for cloud)
      - OLLAMA_MODEL (default: gpt-oss:120b)
    """

    def __init__(self) -> None:
        self.logger = get_logger("llm")
        self.host = os.getenv("OLLAMA_HOST", "https://ollama.com")
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
        self.api_key = os.getenv("OLLAMA_API_KEY")
        try:
            from ollama import Client  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("ollama package is required for LLM extraction") from e
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = Client(host=self.host, headers=headers)

    def extract(self, layout: LayoutsData) -> Result:
        lines = self._collect_lines(layout)
        prompt = self._build_prompt(lines)
        self.logger.info("llm: sending %d lines to model=%s", len(lines), self.model)

        try:
            messages = [{"role": "user", "content": prompt}]
            chunks = self.client.chat(self.model, messages=messages, stream=True)
            text_parts: List[str] = []
            for part in chunks:
                try:
                    text_parts.append(part.get("message", {}).get("content", ""))
                except Exception:
                    continue
            raw = "".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"ollama request failed: {e}")

        data = self._json_from_text(raw)
        if data is None:
            raise RuntimeError("LLM did not return valid JSON")
        try:
            return Result.model_validate(data)
        except Exception:
            # Try to map a flat dict into nested Result
            return self._coerce_to_result(data)

    # -------- helpers --------
    def _collect_lines(self, layout: LayoutsData) -> List[str]:
        lines: List[str] = []
        for page in layout.pages:
            blocks = page.blocks
            if page.reading_order:
                id2b = {b.id: b for b in blocks}
                ordered: List[LayoutBlock] = [id2b[i] for i in page.reading_order if i in id2b]
            else:
                ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            for b in ordered:
                lbl = (b.label or "").strip()
                txt = (b.text or "").strip()
                if lbl and txt:
                    lines.append(f"{lbl}: {txt}")
                elif lbl:
                    lines.append(lbl)
                elif txt:
                    lines.append(txt)
        return lines

    def _build_prompt(self, lines: List[str]) -> str:
        schema = {
            "doc_type": "string|null",
            "direction_type": "string|null",
            "source_org": {
                "name": "string|null",
                "address": "string|null",
                "email": "string|null",
                "phone": "string|null",
                "okved": "string|null",
                "ogrn": "string|null",
            },
            "medical_org": {
                "name": "string|null",
                "address": "string|null",
                "ogrn": "string|null",
                "email": "string|null",
                "phone": "string|null",
            },
            "patient": {
                "full_name": "string|null",
                "birth_date": "string|null",
                "gender": "string|null",
                "snils": "string|null",
                "policy_number": "string|null",
            },
            "employment": {
                "status": "string|null",
                "department": "string|null",
                "job_type": "string|null",
                "position": "string|null",
                "experience": "string|null",
                "previous_jobs": "string|null",
            },
            "hazards": {
                "chemicals": "string|null",
                "biological": "string|null",
                "aerosols_dust": "string|null",
                "physical": "string|null",
                "heavy_labor": "string|null",
                "labour_process": "string|null",
                "performed_works": "string|null",
            },
            "errors": [],
        }
        instructions = (
            "Ты — ИИ, который извлекает ключевые поля из распознанного текста документа. "
            "Верни только валидный JSON без комментариев и объяснений, строго по схеме ниже. "
            "Если поле не найдено — ставь null. Даты в формате dd.mm.yyyy или yyyy-mm-dd. "
            "Пол: Муж/Жен.")
        content = "\n".join(lines[:800])  # hard cap to avoid huge prompts
        return (
            f"{instructions}\n\nСхема (ключи и вложенность):\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Текст документа (в порядке чтения):\n{content}\n\nВозвращай ТОЛЬКО JSON."
        )

    def _json_from_text(self, text: str) -> Optional[dict]:
        if not text:
            return None
        # Find first {...} block
        start = text.find("{")
        if start == -1:
            return None
        # Greedy search to last '}'
        last = text.rfind("}")
        if last == -1 or last <= start:
            return None
        snippet = text[start : last + 1]
        # Remove trailing code fences if present
        snippet = re.sub(r"^```(?:json)?", "", snippet.strip())
        snippet = re.sub(r"```$", "", snippet.strip())
        try:
            return json.loads(snippet)
        except Exception:
            # Try to relax quotes or fix minor issues could be added here
            return None

    def _coerce_to_result(self, data: dict) -> Result:
        # Best-effort mapping when model returns slightly different structure
        def get(path: List[str]) -> Optional[str]:
            cur: object = data
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur if isinstance(cur, str) else None

        coerced = {
            "doc_type": data.get("doc_type"),
            "direction_type": data.get("direction_type"),
            "source_org": {
                "name": get(["source_org", "name"]) or data.get("source_org_name"),
                "address": get(["source_org", "address"]) or data.get("source_org_address"),
                "email": get(["source_org", "email"]) or data.get("source_org_email"),
                "phone": get(["source_org", "phone"]) or data.get("source_org_phone"),
                "okved": get(["source_org", "okved"]) or data.get("okved"),
                "ogrn": get(["source_org", "ogrn"]) or data.get("ogrn"),
            },
            "medical_org": {
                "name": get(["medical_org", "name"]) or data.get("medical_org_name"),
                "address": get(["medical_org", "address"]) or data.get("medical_org_address"),
                "ogrn": get(["medical_org", "ogrn"]) or data.get("medical_ogrn"),
                "email": get(["medical_org", "email"]) or data.get("medical_org_email"),
                "phone": get(["medical_org", "phone"]) or data.get("medical_org_phone"),
            },
            "patient": {
                "full_name": get(["patient", "full_name"]) or data.get("patient_name"),
                "birth_date": get(["patient", "birth_date"]) or data.get("birth_date"),
                "gender": get(["patient", "gender"]) or data.get("gender"),
                "snils": get(["patient", "snils"]) or data.get("snils"),
                "policy_number": get(["patient", "policy_number"]) or data.get("policy"),
            },
            "employment": {
                "status": get(["employment", "status"]) or data.get("employment_status"),
                "department": get(["employment", "department"]) or data.get("department"),
                "job_type": get(["employment", "job_type"]) or data.get("job_type"),
                "position": get(["employment", "position"]) or data.get("position"),
                "experience": get(["employment", "experience"]) or data.get("experience"),
                "previous_jobs": get(["employment", "previous_jobs"]) or data.get("previous_jobs"),
            },
            "hazards": {
                "chemicals": get(["hazards", "chemicals"]) or data.get("chemicals"),
                "biological": get(["hazards", "biological"]) or data.get("biological"),
                "aerosols_dust": get(["hazards", "aerosols_dust"]) or data.get("aerosols_dust"),
                "physical": get(["hazards", "physical"]) or data.get("physical"),
                "heavy_labor": get(["hazards", "heavy_labor"]) or data.get("heavy_labor") or data.get("heavy_labour"),
                "labour_process": get(["hazards", "labour_process"]) or data.get("labour_process"),
                "performed_works": get(["hazards", "performed_works"]) or data.get("performed_works"),
            },
            "errors": data.get("errors", []),
        }
        return Result.model_validate(coerced)
