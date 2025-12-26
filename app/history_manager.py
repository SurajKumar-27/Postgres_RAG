from sqlalchemy.orm import Session
from app.database import ChatSession, ChatMessage
import uuid

def get_or_create_session(db: Session, session_id: str = None) -> str:
    """Returns a valid session_id. Creates a new one if missing or invalid."""
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            return session.id
    
    # Create new session
    new_session = ChatSession(id=str(uuid.uuid4()))
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session.id

def add_message(db: Session, session_id: str, role: str, content: str):
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()

def get_chat_history(db: Session, session_id: str, limit: int = 10):
    """Fetches last N messages for context."""
    return db.query(ChatMessage)\
             .filter(ChatMessage.session_id == session_id)\
             .order_by(ChatMessage.created_at.asc())\
             .limit(limit)\
             .all()