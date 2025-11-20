import os
import time
import google.generativeai as genai
from sqlalchemy import create_engine, inspect, text
from app.database import SessionLocal, SchemaEmbedding, create_vector_tables

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_EXCLUDE_LIST = {"schema_embeddings", "alembic_version"}
# DELAY_SECONDS = 5  # Small delay between embedding calls to avoid 429
EMBEDDING_MODEL = "models/embedding-004"
# ---------------------

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not set. Get one from https://aistudio.google.com/app/apikey")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
db = SessionLocal()

# --- Embedding function (Gemini) ---
def get_gemini_embedding(text: str):
    """Generate an embedding using Gemini with rate-limit handling."""
    while True:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"  # can also be 'retrieval_query'
            )
            embedding = result["embedding"]
            print("✅ Gemini embedding generated successfully.")
            # time.sleep(DELAY_SECONDS)  # pace requests
            return embedding
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            # print(f"⏳ Waiting {DELAY_SECONDS} seconds before retry...")
            # time.sleep(DELAY_SECONDS)

# --- Fetch table/column comments from PostgreSQL ---
def get_table_comments():
    """Fetch COMMENT descriptions for tables and columns."""
    query = text("""
        SELECT 
            c.relname AS table_name,
            a.attname AS column_name,
            d.description
        FROM 
            pg_class c
            JOIN pg_attribute a ON a.attrelid = c.oid
            LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = a.attnum
        WHERE 
            c.relkind = 'r'
            AND a.attnum > 0
            AND NOT a.attisdropped
            AND c.relname NOT LIKE 'pg_%'
            AND c.relname NOT LIKE 'sql_%'
        UNION
        SELECT 
            c.relname AS table_name,
            NULL AS column_name,
            d.description
        FROM 
            pg_class c
            LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
        WHERE 
            c.relkind = 'r'
            AND c.relname NOT LIKE 'pg_%'
            AND c.relname NOT LIKE 'sql_%';
    """)
    comments = {}
    with engine.connect() as conn:
        result = conn.execute(query)
        for table, column, desc in result:
            if desc:
                comments.setdefault(table, {})
                if column:
                    comments[table][column] = desc
                else:
                    comments[table]["_table_"] = desc
    return comments

# --- Ingestion pipeline ---
def ingest_schema():
    """Extract schema, generate Gemini embeddings, and store them."""
    print("🚀 Starting schema ingestion with Gemini embeddings...")
    create_vector_tables()
    db.query(SchemaEmbedding).delete()

    schema_names = inspector.get_schema_names()
    comments = get_table_comments()
    embeddings_to_add = []

    for schema in schema_names:
        if schema.startswith("pg_") or schema == "information_schema":
            continue

        tables = inspector.get_table_names(schema=schema)
        for table_name in tables:
            if table_name in TABLE_EXCLUDE_LIST:
                continue

            print(f"\n📘 Processing table: {table_name}")

            # --- Table-level embedding ---
            table_comment = comments.get(table_name, {}).get("_table_", f"Table named {table_name}")
            table_text = f"Table: {table_name}. Description: {table_comment}"
            table_embedding = get_gemini_embedding(table_text)

            embeddings_to_add.append(SchemaEmbedding(
                table_name=table_name,
                description=table_text,
                embedding=table_embedding
            ))

            # --- Column-level embeddings ---
            columns = inspector.get_columns(table_name, schema=schema)
            for column in columns:
                col_name = column["name"]
                col_type = str(column["type"])
                col_comment = comments.get(table_name, {}).get(col_name, f"Column {col_name} of type {col_type}")

                col_text = f"Table: {table_name}. Column: {col_name}. Type: {col_type}. Description: {col_comment}"
                col_embedding = get_gemini_embedding(col_text)

                embeddings_to_add.append(SchemaEmbedding(
                    table_name=table_name,
                    column_name=col_name,
                    description=col_text,
                    embedding=col_embedding
                ))

    db.add_all(embeddings_to_add)
    db.commit()

    print(f"\n✅ Successfully ingested and embedded schema for {len(embeddings_to_add)} items.")
    db.close()

if __name__ == "__main__":
    ingest_schema()
