from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class IngestRequest(BaseModel):
    document_path: str
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = {}
    correlation_id: str = ""


class IngestJobStatus(BaseModel):
    job_id: str
    document_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    correlation_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    chunk_count: Optional[int] = None
