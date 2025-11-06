import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.database import db_engine, SchemaEmbedding

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Get it from https://aistudio.google.com/app/apikey")

# --- 1. LLM Setup ---

# Use Gemini 1.5 Pro for SQL generation (strong reasoning)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

# Use a faster model for final answer generation (optional)
answer_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

# Embedding model for schema retrieval (Gemini's embedding-001)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

# --- 2. Database Setup ---

db = SQLDatabase(db_engine)
VectorSession = sessionmaker(bind=db_engine)

# --- 3. Schema Retriever ---

def get_schema_retriever(query: str, k: int = 20):
    """
    Custom retriever that finds relevant tables/columns for a query.
    Uses Gemini embeddings instead of OpenAI.
    """
    session = VectorSession()
    try:
        query_embedding = embeddings.embed_query(query)

        # Find the top-k most similar schema snippets (requires numeric vector comparison)
        # If you're storing embeddings as JSON, load all and compute cosine similarity manually.
        schemas = session.query(SchemaEmbedding).all()

        # Compute similarity in Python (no pgvector)
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np

        scored = []
        for s in schemas:
            emb = np.array(s.embedding)
            sim = dot(emb, query_embedding) / (norm(emb) * norm(query_embedding))
            scored.append((sim, s))

        top_k = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        table_names = {s.table_name for _, s in top_k}

        if not table_names:
            return ""

        return db.get_table_info(table_names=list(table_names))

    finally:
        session.close()


# --- 4. SQL Generation Chain ---

sql_prompt = PromptTemplate(
    template="""
    You are an expert PostgreSQL analyst. Given a user question, first retrieve the relevant schema,
    then write a syntactically correct PostgreSQL query to answer it.
    
    DO NOT generate queries that modify the database (UPDATE, DELETE, DROP, etc.).
    Only use tables and columns from the provided schema.

    User Question: {input}
    Relevant Schema:
    {schema}

    SQL Query:
    """,
    input_variables=["input", "schema"]
)

sql_generation_chain = (
    RunnablePassthrough.assign(schema=lambda x: get_schema_retriever(x["input"]))
    | sql_prompt
    | llm
    | StrOutputParser()
)


# --- 5. Query Execution ---

def run_sql_query(query: str):
    """
    Runs the SQL query safely after cleaning markdown formatting.
    """
    try:
        # 🧹 Remove markdown code fences and language tags (like ```sql)
        cleaned_query = re.sub(r"```[a-zA-Z]*\n?", "", query).strip("` \n")

        print(f"🧾 Executing cleaned SQL:\n{cleaned_query}")

        return db.run(cleaned_query)
    except Exception as e:
        return f"Error: {e}"


# --- 6. Answer Generation Chain ---

answer_prompt = PromptTemplate(
    template="""
    Given a user question, the SQL query you generated, and the result of that query,
    formulate a concise, natural language answer.

    User Question: {input}
    SQL Query: {query}
    SQL Result: {result}

    Answer:
    """,
    input_variables=["input", "query", "result"]
)

full_chat_chain = (
    RunnablePassthrough.assign(query=sql_generation_chain)
    | RunnablePassthrough.assign(result=lambda x: run_sql_query(x["query"]))
    | answer_prompt
    | answer_llm
    | StrOutputParser()
)


# --- 7. Main Function ---

def get_chat_response(query: str) -> dict:
    """
    Main entry point for the SQL RAG agent using Gemini.
    """
    response = full_chat_chain.invoke({"input": query})
    generated_sql = sql_generation_chain.invoke({"input": query})
    return {
        "answer": response,
        "generated_sql": generated_sql
    }
