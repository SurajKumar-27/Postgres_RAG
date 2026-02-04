import logging
from io import BytesIO
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from reportlab.pdfgen import canvas

from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    ExportRequest,
    DocumentQuestionRequest,
)
from app.core import get_chat_response, get_document_answer
from app.database import SessionLocal
# Import our new helpers
from app.history_manager import get_or_create_session, add_message, get_chat_history
from app.utils import extract_text_from_pdf, extract_text_from_excel

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Postgres RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENT_STORE = {}

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/chat", response_model=ChatResponse)
def chat_with_db(request: ChatRequest, db: Session = Depends(get_db)):
    logger.info("Received query: %s", request.query)
    
    try:
        # 1. Manage Session (Get existing or create new)
        session_id = get_or_create_session(db, request.session_id)
        
        # 2. Retrieve History from DB
        # Only fetch the last 6 messages to keep tokens low, but enough for context
        db_history = get_chat_history(db, session_id, limit=6)
        for msg in db_history:
            print({
                "id": msg.id,
                "role": msg.role,
                "message": msg.content,
                "created_at": msg.created_at
            })

        
        # 3. Save the *Current* User Query to DB
        add_message(db, session_id, "user", request.query)

        # 4. Run Core Logic (Pass history objects to core)
        # Note: core.py expects objects with .role and .content attributes, 
        # which our SQLAlchemy models already have.
        res = get_chat_response(request.query, history=db_history)

        # Construct a rich log for the history
        assistant_content = res.get("answer")
        if res.get("generated_sql"):
            assistant_content += f"\n\nContext SQL:\n{res.get('generated_sql')}"

        # Save this combined version to DB
        add_message(db, session_id, "assistant", assistant_content)

        return ChatResponse(
            answer=res.get("answer"),
            session_id=session_id,  # Return ID so frontend knows context
            suggested_questions=res.get("suggested_questions", []),
            generated_sql=res.get("generated_sql"),
            backend_raw=res.get("backend_raw"),
            mode=res.get("mode")
        )

    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    filename = file.filename
    extension = filename.split(".")[-1].lower()
    content = file.file.read()

    if extension == "pdf":
        extracted_text, pages_or_sheets = extract_text_from_pdf(content)
        file_type = "pdf"
    elif extension in {"xlsx", "xls"}:
        extracted_text, pages_or_sheets = extract_text_from_excel(content)
        file_type = "excel"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    document_id = str(uuid4())
    DOCUMENT_STORE[document_id] = {
        "filename": filename,
        "file_type": file_type,
        "content": extracted_text,
    }

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        pages_or_sheets=pages_or_sheets,
        extracted_characters=len(extracted_text),
    )


@app.post("/documents/{document_id}/ask")
def ask_document_question(document_id: str, request: DocumentQuestionRequest):
    document = DOCUMENT_STORE.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    answer = get_document_answer(request.question, document["content"])
    return {"answer": answer, "document_id": document_id}


@app.post("/export/pdf")
def export_pdf(request: ExportRequest):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setTitle(request.title)
    pdf.drawString(72, 800, request.title)
    text_object = pdf.beginText(72, 780)
    for line in request.content.splitlines():
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    headers = {"Content-Disposition": f"attachment; filename={request.title}.pdf"}
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)


@app.post("/export/excel")
def export_excel(request: ExportRequest):
    rows = request.rows or [{"content": request.content}]
    dataframe = pd.DataFrame(rows)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Results")
    buffer.seek(0)

    headers = {"Content-Disposition": f"attachment; filename={request.title}.xlsx"}
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )
