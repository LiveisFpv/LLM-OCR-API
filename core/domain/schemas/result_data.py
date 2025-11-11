from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .layout import LayoutsData


class ErrorEntry(BaseModel):
    code: Optional[str] = None
    message: str
    field: Optional[str] = None
    source_block: Optional[str] = None


class OrganizationInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    okved: Optional[str] = None
    ogrn: Optional[str] = None


class MedicalOrganizationInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    ogrn: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class PatientInfo(BaseModel):
    full_name: Optional[str] = None
    birth_date: Optional[str] = None  # ISO yyyy-mm-dd or dd.mm.yyyy as extracted
    gender: Optional[str] = None
    snils: Optional[str] = None
    policy_number: Optional[str] = None


class EmploymentInfo(BaseModel):
    status: Optional[str] = None  # Работающий/Поступающий на работу
    department: Optional[str] = None
    job_type: Optional[str] = None
    position: Optional[str] = None
    experience: Optional[str] = None
    previous_jobs: Optional[str] = None


class HazardsInfo(BaseModel):
    chemicals: Optional[str] = None
    biological: Optional[str] = None
    aerosols_dust: Optional[str] = None
    physical: Optional[str] = None
    heavy_labor: Optional[str] = None
    labour_process: Optional[str] = None
    performed_works: Optional[str] = None


class MetaInfo(BaseModel):
    request_id: Optional[str] = None
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class Result(BaseModel):
    # high-level document type if you classify it (optional)
    doc_type: Optional[str] = None
    direction_type: Optional[str] = None

    source_org: OrganizationInfo = Field(default_factory=OrganizationInfo)
    medical_org: MedicalOrganizationInfo = Field(default_factory=MedicalOrganizationInfo)
    patient: PatientInfo = Field(default_factory=PatientInfo)
    employment: EmploymentInfo = Field(default_factory=EmploymentInfo)
    hazards: HazardsInfo = Field(default_factory=HazardsInfo)

    errors: List[ErrorEntry] = Field(default_factory=list)


class ResultData(BaseModel):
    meta: MetaInfo = Field(default_factory=MetaInfo)
    layout_ir: Optional[LayoutsData] = None
    result: Result = Field(default_factory=Result)
