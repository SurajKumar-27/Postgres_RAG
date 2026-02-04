import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Text, String, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

db_engine = create_engine(DB_URL)

class Base(DeclarativeBase):
    pass

# --- Existing SchemaEmbedding ---
class SchemaEmbedding(Base):
    __tablename__ = "schema_embeddings"
    id = Column(Integer, primary_key=True)
    table_name = Column(Text, nullable=False)
    column_name = Column(Text, nullable=True)
    description = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    __table_args__ = (Index("ix_schema_embeddings_table_name", "table_name"),)

# --- 🆕 NEW: Chat History Tables ---
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSession", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    pages_or_sheets = Column(Integer, nullable=False)
    extracted_characters = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Session Factory ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

def create_tables():
    with db_engine.begin() as conn:
        Base.metadata.create_all(conn)

if __name__ == "__main__":
    create_tables()
