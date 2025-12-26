from typing import Optional, Any, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # 🆕 Frontend sends ID if continuing a chat

class ChatResponse(BaseModel):
    answer: str
    session_id: str  # 🆕 Return ID so frontend can store it
    suggested_questions: List[str] = []
    generated_sql: Optional[str] = None
    backend_raw: Optional[Any] = None
    mode: str