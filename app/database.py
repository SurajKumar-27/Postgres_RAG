import os
from sqlalchemy import create_engine, Column, Integer, Text, Index, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Main Database Connection ---
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Create engine
db_engine = create_engine(DB_URL)


# --- Base model class ---
class Base(DeclarativeBase):
    pass


# --- SchemaEmbedding table definition (no pgvector) ---
class SchemaEmbedding(Base):
    """
    Model to store embeddings for schema tables and columns.
    The embeddings are stored as JSON arrays instead of pgvector.
    """
    __tablename__ = "schema_embeddings"

    id = Column(Integer, primary_key=True)
    table_name = Column(Text, nullable=False)
    column_name = Column(Text, nullable=True)  # Null if it's a table description
    description = Column(Text, nullable=False)  # The text we embedded
    embedding = Column(JSON, nullable=False)  # ✅ Store as JSON instead of Vector

    # Optional: create an index on table_name for faster filtering
    __table_args__ = (
        Index("ix_schema_embeddings_table_name", "table_name"),
    )


# --- SQLAlchemy session ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)



def create_vector_tables():
    """Creates the schema_embeddings table if it doesn't exist."""
    with db_engine.begin() as conn:
        Base.metadata.create_all(conn)


if __name__ == "__main__":
    print("Creating schema_embeddings table in the database...")
    create_vector_tables()
    print("✅ Done.")
