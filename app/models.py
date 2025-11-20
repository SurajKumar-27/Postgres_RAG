# app/models.py
from typing import Optional, Any
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    generated_sql: Optional[str] = None
    backend_raw: Optional[Any] = None
    mode: str  # "sql" or "backend"
