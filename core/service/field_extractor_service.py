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
    """Rule-based extractor tuned for OCR/text output of this form.

    Key points:
    - Reconstruct numbered and labelled lines from layout blocks so that
      downstream parsing sees the original "N. Key: Value" structure even if
      the layout normalizer split key/value.
    - Robust Russian keyword detection for major sections.
    - Targeted extraction for organization and medical fields.
    """

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)(?:[()\s-]*\d){8,12}(?!\d)")
    OGRN_RE = re.compile(r"\b\d{13}\b")
    OKVED_RE = re.compile(r"\b\d{2}\.\d{2}(?:\.\d{1,2})?\b")

    def extract(self, layout: LayoutsData) -> Result:
        logger = get_logger("extract")

        lines = self._collect_lines(layout)

        # Indices of major sections
        heading_idx = self._find_index(
            lines, lambda l: re.search(r"направление\s+на\s+.*медицинск.*осмотр", l, re.I) is not None
        )
        med_idx = self._find_index(lines, lambda l: re.match(r"^\s*направляется\b", l, re.I) is not None)
        patient_idx = self._find_index(lines, lambda l: re.match(r"^\s*1\.\s", l) is not None)

        source_lines = lines[:heading_idx]
        medical_lines = lines[med_idx:patient_idx] if med_idx < len(lines) and med_idx < patient_idx else []
        numbered_lines = lines[patient_idx:] if patient_idx < len(lines) else []

        source_org = self._parse_source_org(source_lines)
        medical_org = self._parse_medical_org(medical_lines)
        patient, employment, hazards = self._parse_patient_and_sections(numbered_lines)

        # Direction type (предварительный/периодический/внеочередной)
        direction_type = None
        if heading_idx < len(lines):
            h = lines[heading_idx].lower()
            if "предварительн" in h:
                direction_type = "ПРЕДВАРИТЕЛЬНЫЙ"
            elif "периодическ" in h:
                direction_type = "ПЕРИОДИЧЕСКИЙ"
            elif "внеочередн" in h:
                direction_type = "ВНЕОЧЕРЕДНОЙ"

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

        self._dump_tmp_result(result)
        logger.info(
            "extracted: org=%s; patient=%s; medical_org=%s",
            result.source_org.name,
            result.patient.full_name,
            result.medical_org.name,
        )
        return result

    # ------------------------------------------------------------------
    # Parsing helpers

    def _collect_lines(self, layout: LayoutsData) -> List[str]:
        """Collect text lines from layout, reconstructing labels if present.

        LayoutService may store label in `block.label` and put only the value in
        `block.text` for meta key-value blocks. For numbered sections we also may
        have `label` with indexes like "9.4". To make downstream parsing robust,
        reconstruct a textual line that contains both label and value.
        """
        lines: List[str] = []
        for page in layout.pages:
            block_map: Dict[str, LayoutBlock] = {block.id: block for block in page.blocks}
            if page.reading_order:
                order = [block_map[bid] for bid in page.reading_order if bid in block_map]
            else:
                order = sorted(page.blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            for block in order:
                txt = (block.text or "").strip()
                lbl = (block.label or "").strip()
                btype = (block.type or "").lower()
                if lbl:
                    if btype == "meta_kv":
                        composed = f"{lbl}: {txt}" if txt else lbl
                    elif btype in {"hazard_item", "section_item", "list_item"} or re.match(r"^\d+(?:\.\d+)?$", lbl):
                        composed = f"{lbl}. {txt}" if txt else f"{lbl}."
                    else:
                        composed = f"{lbl} {txt}".strip()
                    if composed:
                        lines.append(composed)
                        continue
                if txt:
                    lines.append(txt)
        return lines

    def _find_index(self, items: List[str], predicate) -> int:
        for idx, val in enumerate(items):
            try:
                if predicate(val):
                    return idx
            except Exception:
                continue
        return len(items)

    # -------- source organisation --------
    def _parse_source_org(self, lines: List[str]) -> OrganizationInfo:
        text_block = " \n".join(lines)
        email = self._first_match(self.EMAIL_RE, text_block)
        phone = self._pick_best_phone(lines)
        ogrn = self._first_match(self.OGRN_RE, text_block)

        # Name: prefer quoted line or a line containing ООО/Общество с ограниченной ответственностью
        name: Optional[str] = None
        for i, line in enumerate(lines):
            low = line.lower()
            if re.search(r"\bооо\b|обществ[оа] с ограниченн", low):
                # Next line may contain official name in quotes
                next_line = lines[i + 1] if i + 1 < len(lines) else None
                if next_line and (next_line.strip().startswith('"') or '«' in next_line):
                    name = self._norm_name(next_line)
                else:
                    name = self._norm_name(line)
                break
            if line.strip().startswith('"') or '«' in line:
                name = self._norm_name(line)
                break

        # Address: take address-like lines, stop before heading/OGRN label
        address_parts: List[str] = []
        for line in lines:
            if self.EMAIL_RE.search(line) or self.OGRN_RE.search(line):
                continue
            low = line.lower()
            if re.match(r"^\s*\d{5,6}[, ]", line) or any(
                kw in low for kw in [
                    "россия",
                    "округ",
                    "г ",
                    "ул ",
                    "улиц",
                    "д ",
                    "корп",
                    "оф",
                ]
            ):
                address_parts.append(line.strip(" ;,.") )
        address = ", ".join(self._dedupe_preserve(address_parts)) or None

        # Fallback: reconstruct full name from legal form + quoted line
        if not name:
            q_idx = None
            q_line = None
            for i, ln in enumerate(lines):
                s = ln.strip()
                if s.startswith('"') or ("«" in s and "»" in s):
                    q_idx = i
                    q_line = s
                    break
            if q_line is not None:
                legal_form = lines[q_idx - 1].strip(" .,") if q_idx and q_idx > 0 else None
                if legal_form:
                    name = self._norm_name(f"{legal_form} {q_line}")
                else:
                    name = self._norm_name(q_line)

        return OrganizationInfo(
            name=name,
            address=address,
            email=email,
            phone=phone,
            okved=self._first_match(self.OKVED_RE, text_block),
            ogrn=ogrn,
        )

    # -------- medical organisation --------
    def _parse_medical_org(self, lines: List[str]) -> MedicalOrganizationInfo:
        if not lines:
            return MedicalOrganizationInfo()

        joined = " \n".join(lines)
        email = self._first_match(self.EMAIL_RE, joined)
        phone = self._pick_best_phone(lines, prefer_tel=True)
        # Take the last 13-digit number as OGRN (often appears after phone)
        ogrns = self.OGRN_RE.findall(joined)
        ogrn = ogrns[-1] if ogrns else None

        # Name in the first line after the word "Направляется"
        first_line = lines[0]
        name = None
        m = re.match(r"^\s*направляется\s+(.+?)\s*[;,:]", first_line, re.I)
        if m:
            name = self._norm_name(m.group(1))

        # Address: everything after ';' in the first line + follow-up lines without emails/phones/ogrn
        address_parts: List[str] = []
        if ";" in first_line:
            rest = first_line.split(";", 1)[1].strip(" ;,")
            if rest:
                address_parts.append(rest)
        for line in lines[1:]:
            cleaned = self.EMAIL_RE.sub("", line)
            cleaned = self.PHONE_RE.sub("", cleaned)
            cleaned = self.OGRN_RE.sub("", cleaned)
            cleaned = cleaned.strip(" ;,.")
            if not cleaned:
                continue
            if re.match(r"^\d{5,6}[, ]", cleaned, re.I) or re.search(
                r"россия|округ|г\s|ул\s|улиц|д\.|д\s|корп|оф\.", cleaned, re.I
            ):
                address_parts.append(cleaned)
        address = ", ".join(self._dedupe_preserve(address_parts)) or None

        return MedicalOrganizationInfo(
            name=name,
            address=address,
            ogrn=ogrn,
            email=email,
            phone=phone,
        )

    # -------- patient + employment + hazards --------
    def _parse_patient_and_sections(
        self, lines: List[str]
    ) -> tuple[PatientInfo, EmploymentInfo, HazardsInfo]:
        sections = self._collect_numbered_sections(lines)

        def value(num: str) -> Optional[str]:
            text = sections.get(num)
            if not text:
                return None
            return self._value_after_colon(text)

        patient = PatientInfo(
            full_name=value("1"),
            birth_date=value("2"),
            gender=value("3"),
            snils=self._normalize_snils(value("4")),
            policy_number=value("5"),
        )

        employment = EmploymentInfo(
            status=value("6"),
            department=value("7"),
            job_type=None,
            position=value("8"),
            experience=None,
            previous_jobs=None,
        )

        hazards = HazardsInfo(
            chemicals=value("9.1"),
            biological=value("9.2"),
            aerosols_dust=value("9.3"),
            physical=value("9.4"),
            labour_process=value("9.5"),
            performed_works=value("9.6"),
        )

        return patient, employment, hazards

    def _collect_numbered_sections(self, lines: List[str]) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\.\s*(.*)$", stripped)
            if m:
                if current_key is not None:
                    sections[current_key] = " ".join(current_lines).strip()
                current_key = m.group(1)
                rest = m.group(2).strip()
                current_lines = [rest] if rest else []
            else:
                if current_key is None:
                    continue
                # Skip helper comments like "(номер пункта...)"
                if stripped.startswith("(") and stripped.endswith(")"):
                    continue
                if re.search(r"\b[мm]\.п\.?\b|ф\.и\.о|подпись|уполномоченного", stripped, re.I):
                    if current_key is not None and current_lines:
                        sections[current_key] = " ".join(current_lines).strip()
                    break
                current_lines.append(stripped)
        if current_key is not None and current_lines:
            sections[current_key] = " ".join(current_lines).strip()
        return sections

    # -------- utilities --------
    def _first_match(self, pattern: re.Pattern[str], text: str) -> Optional[str]:
        m = pattern.search(text)
        return m.group(0) if m else None

    def _pick_best_phone(self, lines: List[str], prefer_tel: bool = False) -> Optional[str]:
        candidates: List[str] = []
        for line in lines:
            if prefer_tel and re.search(r"тел\.?\s*\+?\d", line.lower()):
                candidates.extend(self.PHONE_RE.findall(line))
            else:
                candidates.extend(self.PHONE_RE.findall(line))
        if not candidates:
            joined = " ".join(lines)
            candidates = self.PHONE_RE.findall(joined)
        if not candidates:
            return None
        candidates = sorted({self._clean_phone(c) for c in candidates}, key=len, reverse=True)
        return candidates[0]

    def _clean_phone(self, phone: str) -> str:
        # Strip trailing extensions in parentheses like "(102,103)"
        phone = re.sub(r"\s*\([^)]*\)\s*$", "", phone)
        phone = re.sub(r"[\s]+", " ", phone)
        # Trim trailing punctuation
        phone = phone.strip(" .,;:")
        return phone.strip()

    def _norm_name(self, name: str) -> str:
        name = name.strip()
        name = re.sub(r"\b[OО]0[OО]\b", "ООО", name, flags=re.I)
        name = re.sub(r"\bОбшество\b", "Общество", name, flags=re.I)
        name = re.sub(r"\s+", " ", name)
        return name

    def _normalize_company_name(self, name: str) -> str:
        name = name.strip(' \"«»')
        # Tolerate OCR zero/letter confusion in ООО
        name = re.sub(r"\bО0О\b", "ООО", name, flags=re.I)
        name = re.sub(r"\s+", " ", name)
        return name

    def _normalize_snils(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        digits = re.sub(r"[^0-9]", "", value)
        if len(digits) == 11:
            return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:11]}"
        return value.strip()

    def _value_after_colon(self, text: str) -> Optional[str]:
        if not text:
            return None
        parts = text.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip(" .;,") or None
        return text.strip() or None

    def _dedupe_preserve(self, items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            key = item.strip().lower()
            if key in seen or not item.strip():
                continue
            seen.add(key)
            result.append(item.strip())
        return result

    def _dump_tmp_result(self, result: Result) -> None:
        try:
            os.makedirs("tmp", exist_ok=True)
            from core.domain.schemas.result_data import ResultData, MetaInfo

            rd = ResultData(meta=MetaInfo(timings_ms={}), layout_ir=None, result=result)
            with open("tmp/result.json", "w", encoding="utf-8") as fh:
                json.dump(rd.model_dump(), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass



