import os
import json
import time
import logging
# 1. Update Imports: Use the consolidated LangChain class for consistency
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy import create_engine, inspect, text
from app.database import SessionLocal, SchemaEmbedding, create_tables

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
            rule_text = f"RULE for {table_name}: {rule}"

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


def ingest_schema():
    print(f"🚀 Starting schema ingestion via Vertex AI ({LOCATION})...")
    create_tables()
    db.query(SchemaEmbedding).delete() # Clear old data
    
    schema_names = inspector.get_schema_names()
    custom_rules = load_custom_rules()
    embeddings_to_add = []

    for schema in schema_names:
        if schema.startswith("pg_") or schema == "information_schema": continue
        
        tables = inspector.get_table_names(schema=schema)
        for table_name in tables:
            if table_name in TABLE_EXCLUDE_LIST: continue
            
            print(f"📘 Processing {table_name}...")

            # 1. Embed Table Description
            table_text = f"Table: {table_name}."
            embeddings_to_add.append(SchemaEmbedding(
                table_name=table_name, 
                description=table_text, 
                embedding=get_gemini_embedding(table_text)
            ))

            # 2. Auto-Embed Foreign Keys
            try:
                fks = inspector.get_foreign_keys(table_name, schema=schema)
                for fk in fks:
                    fk_text = (f"RELATIONSHIP: {table_name}.{', '.join(fk['constrained_columns'])} "
                               f"joins to {fk['referred_table']}.{', '.join(fk['referred_columns'])}.")
                    
                    embeddings_to_add.append(SchemaEmbedding(
                        table_name=table_name,
                        description=fk_text,
                        embedding=get_gemini_embedding(fk_text)
                    ))
            except Exception as e:
                print(f"   Warning reading FKs for {table_name}: {e}")

            # 3. Embed Custom Rules
            if table_name in custom_rules:
                for rule in custom_rules[table_name]:
                    rule_text = f"RULE for {table_name}: {rule}"
                    embeddings_to_add.append(SchemaEmbedding(
                        table_name=table_name,
                        description=rule_text,
                        embedding=get_gemini_embedding(rule_text)
                    ))

            # 4. Embed Columns
            cols = inspector.get_columns(table_name, schema=schema)
            for c in cols:
                col_text = f"Column: {table_name}.{c['name']} ({c['type']})."
                embeddings_to_add.append(SchemaEmbedding(
                    table_name=table_name, column_name=c['name'],
                    description=col_text, embedding=get_gemini_embedding(col_text)
                ))

    db.add_all(embeddings_to_add)
    db.commit()
    print(f"✅ Successfully embedded {len(embeddings_to_add)} schema items using text-embedding-004.")
    db.close()

if __name__ == "__main__":
    import sys

    if "--rules-only" in sys.argv:
        ingest_business_rules_only()
    else:
        ingest_schema()
