import os
import json
import time
import google.generativeai as genai
from sqlalchemy import create_engine, inspect, text
from app.database import SessionLocal, SchemaEmbedding, create_tables

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_EXCLUDE_LIST = {"schema_embeddings", "alembic_version"}
EMBEDDING_MODEL = "models/gemini-embedding-001"
RULES_FILE = "C:/Users/suraj.marepally\OneDrive - The Hackett Group, Inc/sqlRag/rag/schema_rules.json"  # Path to your rules file

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not set.")

genai.configure(api_key=GEMINI_API_KEY)
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
db = SessionLocal()

def get_gemini_embedding(text: str):
    """Generates embedding with retry logic."""
    while True:
        try:
            return genai.embed_content(
                model=EMBEDDING_MODEL, content=text, task_type="retrieval_document"
            )["embedding"]
        except Exception as e:
            print(f"⚠️ API error: {e}. Retrying in 2s...")
            time.sleep(2)

def load_custom_rules():
    """Loads specific business rules from JSON file."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r") as f:
            return json.load(f)
    return {}

def ingest_schema():
    print("🚀 Starting schema ingestion...")
    create_tables()
    db.query(SchemaEmbedding).delete() # Clear old data
    
    schema_names = inspector.get_schema_names()
    custom_rules = load_custom_rules()
    print(custom_rules)
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

            # 2. Auto-Embed Foreign Keys (Scales to all 70 tables)
            try:
                fks = inspector.get_foreign_keys(table_name, schema=schema)
                for fk in fks:
                    # Create a clear rule string for the LLM
                    fk_text = (f"RELATIONSHIP: {table_name}.{', '.join(fk['constrained_columns'])} "
                               f"joins to {fk['referred_table']}.{', '.join(fk['referred_columns'])}.")
                    
                    embeddings_to_add.append(SchemaEmbedding(
                        table_name=table_name,
                        description=fk_text,
                        embedding=get_gemini_embedding(fk_text)
                    ))
            except Exception as e:
                print(f"   Warning reading FKs for {table_name}: {e}")

            # 3. Embed Custom Rules from JSON
            if table_name in custom_rules:
                for rule in custom_rules[table_name]:
                    rule_text = f"RULE for {table_name}: {rule}"
                    embeddings_to_add.append(SchemaEmbedding(
                        table_name=table_name,
                        description=rule_text,
                        embedding=get_gemini_embedding(rule_text)
                    ))

            # 4. Embed Columns (Standard)
            cols = inspector.get_columns(table_name, schema=schema)
            for c in cols:
                col_text = f"Column: {table_name}.{c['name']} ({c['type']})."
                embeddings_to_add.append(SchemaEmbedding(
                    table_name=table_name, column_name=c['name'],
                    description=col_text, embedding=get_gemini_embedding(col_text)
                ))

    db.add_all(embeddings_to_add)
    db.commit()
    print(f"✅ Successfully embedded {len(embeddings_to_add)} schema items.")
    db.close()

if __name__ == "__main__":
    ingest_schema()