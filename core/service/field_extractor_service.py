from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from core.lib.logger import get_logger
from core.domain.schemas.layout import LayoutBlock, LayoutsData
from core.domain.schemas.result_data import (
    EmploymentInfo,
    HazardsInfo,
    MedicalOrganizationInfo,
    OrganizationInfo,
    PatientInfo,
    Result,
)
from core.domain.ports.Field_extractor_provider import Field_extractor_provider


class FieldExtractorService(Field_extractor_provider):
    """Extractor that uses structured layout blocks with regex fallbacks.

    - Consumes blocks of type: text, meta_kv, hazard_item, heading.
    - Builds a Result with organization, patient, employment, hazards.
    - Saves JSON dump into tmp/result.json for quick inspection.
    """

    def extract(self, layout: LayoutsData) -> Result:
        logger = get_logger("extract")

        texts: List[str] = []
        kv_map: Dict[str, str] = {}
        hazards_by_idx: Dict[str, str] = {}

        for p in layout.pages:
            for b in p.blocks:
                if b.text:
                    texts.append(b.text)
                if b.type == "meta_kv" and b.label and b.text:
                    kv_map.setdefault(b.label.lower(), b.text)
                if b.type == "hazard_item" and b.label and b.text:
                    hazards_by_idx[b.label] = b.text

        full_text = "\n".join(texts)

        source_org = OrganizationInfo(
            name=self._find_kv(kv_map, full_text, ["наименование организации", "наименование организации (предприятия)"]),
            address=self._find_kv(kv_map, full_text, ["адрес", "адрес регистрации"]),
            email=self._regex(full_text, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            phone=self._regex(full_text, r"(?:\+7|8)[\s\-\(]?\d{3}[\)\s\-]?\s?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
            okved=self._regex(full_text, r"оквэд[^\d]*([0-9]{2}\.[0-9]{2}(?:\.[0-9]{1,2})?)", group=1, flags=re.I),
            ogrn=self._regex_near(full_text, r"огрн|код огрн", r"\b\d{13}\b"),
        )

        medical_org = MedicalOrganizationInfo(
            name=self._find_kv(kv_map, full_text, ["наименование медицинской организации", "направляется в"]),
            address=self._find_kv(kv_map, full_text, ["адрес регистрации", "адрес"]),
            ogrn=self._regex_near(full_text, r"огрн|код огрн", r"\b\d{13}\b"),
            email=self._regex(full_text, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            phone=self._regex(full_text, r"(?:\+7|8)[\s\-\(]?\d{3}[\)\s\-]?\s?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"),
        )

        patient = PatientInfo(
            full_name=self._find_kv(kv_map, full_text, ["ф. и. о.", "фио", "ф.и.о."]) or self._regex(full_text, r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b"),
            birth_date=self._regex_near(full_text, r"дата рождения", r"\b[0-3]\d\.[01]\d\.\d{4}\b"),
            gender=self._find_kv(kv_map, full_text, ["пол"]),
            snils=self._regex_near(full_text, r"снилс", r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"),
            policy_number=self._regex_near(full_text, r"номер полиса|омс|дмс", r"\b[0-9\s\-]{8,}\b"),
        )

        employment = EmploymentInfo(
            status=self._regex(full_text, r"(поступающий на работу|работающий)", flags=re.I),
            department=self._find_kv(kv_map, full_text, ["цех", "участок", "структурное подразделение"]),
            job_type=self._find_kv(kv_map, full_text, ["вид работы"]),
            position=self._find_kv(kv_map, full_text, ["наименование должности", "профессия", "должность/профессия"]),
            experience=self._find_kv(kv_map, full_text, ["стаж работы"]),
            previous_jobs=self._find_kv(kv_map, full_text, ["предшествующие профессии", "предшествующие профессии (работы)"]),
        )

        hazards = HazardsInfo(
            chemicals=hazards_by_idx.get("8.1") or hazards_by_idx.get("9.1") or self._hazard(texts, ["8.1", "9.1"], ["химические факторы"]),
            biological=hazards_by_idx.get("9.2") or self._hazard(texts, ["9.2"], ["биологические факторы"]),
            aerosols_dust=hazards_by_idx.get("9.3") or self._hazard(texts, ["9.3"], ["аэрозоли", "пыли"]),
            physical=hazards_by_idx.get("8.2") or hazards_by_idx.get("9.4") or self._hazard(texts, ["8.2", "9.4"], ["физические факторы"]),
            labour_process=hazards_by_idx.get("9.5") or self._hazard(texts, ["9.5"], ["факторы трудового процесса"]),
            performed_works=hazards_by_idx.get("9.6") or self._hazard(texts, ["9.6"], ["выполняемые работы"]),
        )

        direction_type = self._regex(full_text, r"(предварительный|периодический|повторный)", flags=re.I)

        result = Result(
            doc_type="napravlenie_med_osmotr",
            direction_type=direction_type,
            source_org=source_org,
            medical_org=medical_org,
            patient=patient,
            employment=employment,
            hazards=hazards,
            errors=[],
        )

        # Dump to tmp/result.json
        try:
            os.makedirs("tmp", exist_ok=True)
            out_path = os.path.join("tmp", "result.json")
            from core.domain.schemas.result_data import ResultData, MetaInfo
            rd = ResultData(meta=MetaInfo(timings_ms={}), layout_ir=None, result=result)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(rd.model_dump(), fh, ensure_ascii=False, indent=2)
            logger.info("result json saved: %s", out_path)
        except Exception as exc:
            logger = get_logger("extract")
            logger.warning("failed to save result json: %s", exc)

        return result

    # ---------- helpers ----------
    def _find_kv(self, kv_map: Dict[str, str], text: str, keys: List[str]) -> Optional[str]:
        for k in keys:
            v = kv_map.get(k.lower())
            if v:
                return v
        return self._kv(text, keys)

    def _kv(self, text: str, anchors: List[str]) -> Optional[str]:
        for a in anchors:
            m = re.search(rf"{re.escape(a)}\s*:\s*(.+)", text, flags=re.I)
            if m:
                return m.group(1).strip()
        return None

    def _regex(self, text: str, pattern: str, group: int = 0, flags: int = 0) -> Optional[str]:
        m = re.search(pattern, text, flags)
        return m.group(group) if m else None

    def _regex_near(self, text: str, anchor: str, pattern: str) -> Optional[str]:
        rx = rf"{anchor}[^\n\r]*?({pattern})"
        m = re.search(rx, text, flags=re.I)
        if m:
            return m.group(1)
        m = re.search(pattern, text)
        return m.group(0) if m else None

    def _hazard(self, lines: List[str], indices: List[str], titles: List[str]) -> Optional[str]:
        for idx in indices:
            for ln in lines:
                if ln.strip().startswith(idx):
                    tail = ln.split(idx, 1)[1].strip(" .:")
                    return tail or None
        for t in titles:
            for ln in lines:
                if t.lower() in ln.lower():
                    parts = re.split(r"[:：]", ln, maxsplit=1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
        return None

