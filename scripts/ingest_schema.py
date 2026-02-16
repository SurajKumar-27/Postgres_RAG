import os
import json
import time
import logging
# 1. Update Imports: Use the consolidated LangChain class for consistency
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy import create_engine, inspect, text
from app.database import SessionLocal, SchemaEmbedding, create_tables
from app.utils import execute_remote_query 

# ---------- Configuration ----------
# 2. Update Environment Variables for Vertex AI
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
DATABASE_URL = os.getenv("DATABASE_URL")

TABLE_EXCLUDE_LIST = {"schema_embeddings", "alembic_version","chat_sessions","chat_messages"}
RULES_FILE = "C:/Users/suraj.marepally/OneDrive - The Hackett Group, Inc/sqlRag/rag/schema_rules.json"

if not PROJECT_ID:
    raise ValueError("⚠️ GOOGLE_CLOUD_PROJECT not set for Vertex AI.")

# # 4. Initialize Embeddings Object (mirrors app/core.py)
# embeddings_service = GoogleGenerativeAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     project=PROJECT_ID,
#     location=LOCATION,
#     vertexai=True
# )

embeddings_service = VertexAIEmbeddings(
    model_name="text-embedding-004", # Production-ready embedding model
    project=PROJECT_ID,
    location=LOCATION
)

engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
db = SessionLocal()

def get_gemini_embedding(text: str):
    """Generates embedding using Vertex AI backend with retry logic."""
    while True:
        try:
            # 5. Use the embed_query method from the LangChain class
            return embeddings_service.embed_query(text)
        except Exception as e:
            print(f"⚠️ Vertex AI API error: {e}. Retrying in 2s...")
            time.sleep(2)

def load_custom_rules():
    """Loads specific business rules from JSON file."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r") as f:
            return json.load(f)
    return {}

def ingest_business_rules_only():
    print("🚀 Re-ingesting ONLY business rule embeddings...")

    custom_rules = load_custom_rules()
    if not custom_rules:
        print("ℹ️ No business rules found. Skipping.")
        return

    # 1️⃣ Delete existing RULE embeddings
    deleted = (
        db.query(SchemaEmbedding)
        .filter(SchemaEmbedding.description.like("RULE for %"))
        .delete(synchronize_session=False)
    )
    db.commit()

    print(f"🧹 Deleted {deleted} old rule embeddings")

    embeddings_to_add = []

    # 2️⃣ Re-create embeddings for rules only
    for table_name, rules in custom_rules.items():
        for rule in rules:
            rule_text = f"""
            TABLE: {table_name}
            BUSINESS_RULE: {rule}
            """
            embeddings_to_add.append(
                SchemaEmbedding(
                    table_name=table_name,
                    description=rule_text,
                    embedding=get_gemini_embedding(rule_text)
                )
            )

    # 3️⃣ Insert new rule embeddings
    db.add_all(embeddings_to_add)
    db.commit()

    print(f"✅ Successfully embedded {len(embeddings_to_add)} business rules only.")

# scripts/ingest_schema.py

def ingest_schema():
    print(f"🚀 Starting remote MSSQL schema ingestion via Vertex AI ({LOCATION})...")
    create_tables()
    db.query(SchemaEmbedding).delete() # Clear local PostgreSQL embeddings
    
    # T-SQL to fetch tables and columns from MSSQL
    tsql_query = """
    SELECT 
        t.name AS table_name,
        c.name AS column_name,
        ty.name AS data_type
    FROM sys.tables t
    JOIN sys.columns c ON t.object_id = c.object_id
    JOIN sys.types ty ON c.user_type_id = ty.user_type_id
    WHERE t.is_ms_shipped = 0
    ORDER BY t.name;
    """
    
    try:
        raw_metadata = execute_remote_query(tsql_query)
    except Exception as e:
        print(f"❌ Failed to fetch metadata: {e}")
        return

    custom_rules = load_custom_rules()
    embeddings_to_add = []

    # Grouping metadata by table
    tables_map = {}
    for row in raw_metadata:
        t_name = row['table_name']
        if t_name not in tables_map:
            tables_map[t_name] = []
        tables_map[t_name].append(row)

    for table_name, columns in tables_map.items():
        if table_name in TABLE_EXCLUDE_LIST: 
            continue
            
        print(f"📘 Processing {table_name}...")

        # 1. Embed Table Description
        table_text = f"""
        TABLE: {table_name}
        TYPE: MSSQL_ERP_TABLE
        ROLE: Business transactional or master data table
        """
        embeddings_to_add.append(SchemaEmbedding(
            table_name=table_name, 
            description=table_text, 
            embedding=get_gemini_embedding(table_text)
        ))

        # 2. Embed Custom Business Rules (Modified/Preserved)
        if table_name in custom_rules:
            for rule in custom_rules[table_name]:
                rule_text = f"""
TABLE: {table_name}
BUSINESS_RULE: {rule}
"""
                embeddings_to_add.append(SchemaEmbedding(
                    table_name=table_name,
                    description=rule_text,
                    embedding=get_gemini_embedding(rule_text)
                ))

        # 3. Embed Columns
        for col in columns:
            col_text = f"""
                TABLE: {table_name}
                COLUMN: {col['column_name']}
                TYPE: {col['data_type']}
                ROLE: Field belonging to {table_name}
                """
            embeddings_to_add.append(SchemaEmbedding(
                table_name=table_name,
                column_name=col['column_name'],
                description=col_text.strip(),
                embedding=get_gemini_embedding(col_text)
            ))

    db.add_all(embeddings_to_add)
    db.commit()
    print(f"✅ Successfully embedded {len(embeddings_to_add)} remote items into local PostgreSQL.")
    db.close()

if __name__ == "__main__":
    import sys

    if "--rules-only" in sys.argv:
        ingest_business_rules_only()
    else:
        ingest_schema()
