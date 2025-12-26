import logging
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse
from app.core import get_chat_response
from app.database import SessionLocal
# Import our new helpers
from app.history_manager import get_or_create_session, add_message, get_chat_history

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