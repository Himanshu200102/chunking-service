from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Role = Literal["owner","editor","reader"]

class Member(BaseModel):
    user_id: str
    role: Role

class ProjectIn(BaseModel):
    name: str
    members: List[Member] = []

class ProjectOut(ProjectIn):
    id: str = Field(alias="_id")

class FileIn(BaseModel):
    project_id: str
    filename: str
    mime: Optional[str] = None
    size: Optional[int] = None
    checksum: Optional[str] = None

class FileVersionIn(BaseModel):
    file_id: str
    project_id: str
    version: int = 1
    status: Literal["queued","parsed","chunked","embedded","indexed","failed"] = "queued"
    storage: Dict[str, Optional[str]] = {"raw_uri": None, "docling_json_uri": None, "text_uri": None, "chunks_uri": None}
