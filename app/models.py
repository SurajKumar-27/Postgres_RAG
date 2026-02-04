from typing import Optional, Any, List, Dict
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


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    pages_or_sheets: int
    extracted_characters: int


class ExportRequest(BaseModel):
    title: str
    content: str
    rows: Optional[List[Dict[str, Any]]] = None


class DocumentQuestionRequest(BaseModel):
    question: str


class DocumentAnswerResponse(BaseModel):
    answer: str
    document_id: str
    filename: str
