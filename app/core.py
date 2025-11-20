# app/core.py
import os
import re
import json
import logging
from typing import Any, Dict, Optional, Tuple

import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import sessionmaker
from numpy import dot
from numpy.linalg import norm
import numpy as np

from app.database import db_engine, SchemaEmbedding

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Environment / Config ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Get it from https://aistudio.google.com/app/apikey")

# Optional: token to include in backend requests
BACKEND_AUTH_TOKEN = os.getenv("BACKEND_AUTH_TOKEN", None)

# ---------- 1) LLMs & Embeddings ----------
# Core LLM used for SQL generation / decisioning. Keep temp = 0 for deterministic SQL.
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

# Answer generation LLM (you can pick a faster model)
answer_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)

# Embedding model (schema retrieval)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

# ---------- 2) Database setup ----------
db = SQLDatabase(db_engine)
VectorSession = sessionmaker(bind=db_engine)

# ---------- 3) Schema retriever ----------
def get_schema_retriever(query: str, k: int = 40) -> str:
    """
    Retrieve the most relevant table/column schema snippets for the query by
    comparing embeddings stored in the schema_embeddings table.
    Returns a textual schema summary (or empty string if none found).
    """
    session = VectorSession()
    try:
        query_embedding = embeddings.embed_query(query)
        schemas = session.query(SchemaEmbedding).all()

        scored = []
        for s in schemas:
            emb = np.array(s.embedding)
            sim = dot(emb, query_embedding) / (norm(emb) * norm(query_embedding) + 1e-8)
            scored.append((sim, s))

        top_k = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        table_names = {s.table_name for _, s in top_k}

        if not table_names:
            return ""

        return db.get_table_info(table_names=list(table_names))
    finally:
        session.close()

# ---------- 4) SQL generation chain ----------
sql_prompt = PromptTemplate(
    template="""
You are an expert PostgreSQL analyst. Given a user question and a schema description,
write a single, syntactically correct SELECT SQL query that answers the question,
or return an appropriate read-only query (no UPDATE/DELETE/INSERT/DROP).
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

# ---------- 5) Execute SQL (with auto-retry fix using LLM) ----------
def run_sql_query_with_retry(query: str, question: str):
    """
    Try to run SQL; if it errors, ask the LLM to fix the query and re-run.
    Returns query result (whatever db.run returns) or raises on final failure.
    """
    try:
        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", query).strip("` \n")
        logger.info("Executing SQL:\n%s", cleaned)
        return db.run(cleaned)
    except Exception as e:
        error_msg = str(e)
        logger.warning("SQL execution failed: %s\nInvoking LLM to fix SQL...", error_msg)
        correction_prompt = f"""
The following SQL query failed with an error. Fix it and return only the corrected SQL.

User Question: {question}
SQL Query: {cleaned}
Error: {error_msg}

Return only the corrected SQL query (no explanatory text).
"""
        fixed_sql = llm.invoke(correction_prompt).content
        cleaned_fixed = re.sub(r"```[a-zA-Z]*\n?", "", fixed_sql).strip("` \n")
        logger.info("Running corrected SQL:\n%s", cleaned_fixed)
        return db.run(cleaned_fixed)

# ---------- 6) Backend tool description (for instruction to the LLM) ----------
BACKEND_TOOL_DOC = """
You can call backend REST endpoints by returning a JSON object with the shape:
{
  "tool": "call_backend",
  "arguments": {
    "method": "GET|POST|PUT|DELETE",
    "url": "https://... (full URL)",
    "headers": { "Content-Type": "application/json", "X-Siteid":"200" ... },    
    "body": { ... }                                            
  }
}
Return that JSON (as the only JSON object in your response) when the user intent requires calling an API (create order, update address, call payment service, etc).
If the user's request is purely a data retrieval against the Postgres database, do NOT return the tool JSON. Instead the code will run the SQL pipeline.
"""

# Optional short examples to append to instruction (help LLM generate correct JSON)
BACKEND_TOOL_EXAMPLES = """
Example 1 (order details):
{
 "tool": "call_backend",
 "arguments": {
   "method": "POST",
   "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_regular_order_details_guest",
   "headers": {"Content-Type": "application/json"},
   "body": {"salesOrderNo": "...."}
 }
}

Example 2 (get invoice list):
{
 "tool": "call_backend",
 "arguments": {
   "method": "POST",
   "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_invoice_list",
   "headers": {"Content-Type": "application/json"},
   "body": {"soldTo":"...", "fromDate":"...", "toDate": "..."}
 }
}

Example 3 (invoice details):
{
 "tool": "call_backend",
 "arguments": {
   "method": "POST",
   "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_invoice_details",
   "headers": {"Content-Type": "application/json"},
   "body": {"invoiceNo": "...."}
 }
}

Example 4 (delivery details):
{
 "tool": "call_backend",
 "arguments": {
   "method": "POST",
   "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_delivery_details",
   "headers": {"Content-Type": "application/json"},
   "body": {"deliveryNo": "...."}
 }
}
"""

# ---------- 7) Helpers to parse JSON tool call from model text ----------
def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object inside a text blob.
    Uses brace counting instead of recursive regex (Python-safe).
    Returns a dict or None.
    """

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


# ---------- 8) Backend executor ----------
def execute_backend_call(args: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Executes the backend request described in args and returns (status_code, parsed_body_or_text).
    """
    method = args.get("method", "GET").upper()
    url = args.get("url")
    headers = args.get("headers", {}) or {}
    body = args.get("body", None)

    if not url:
        raise ValueError("Backend tool arguments must include 'url'.")

    # Inject auth header if provided via env
    if BACKEND_AUTH_TOKEN and "Authorization" not in {k.title(): v for k, v in headers.items()}:
        headers["Authorization"] = f"Bearer {BACKEND_AUTH_TOKEN}"

    try:
        resp = requests.request(method=method, url=url, headers=headers, json=body, timeout=30)
    except Exception as e:
        logger.error("Error while calling backend: %s", e, exc_info=True)
        raise

    content_type = resp.headers.get("Content-Type", "")
    try:
        if "application/json" in content_type:
            return resp.status_code, resp.json()
        else:
            # return text for non-json responses
            return resp.status_code, resp.text
    except Exception:
        return resp.status_code, resp.text

# ---------- 9) Backend-answer prompt ----------
backend_answer_prompt = PromptTemplate(
    template="""
You are a helpful assistant. The user asked: {input}

You executed a backend API call and got the following result (raw):
{backend_result}

Provide a concise, actionable, and user-friendly summary of what happened.
If the backend returned an error or non-success status, explain what likely failed and any next steps.
""",
    input_variables=["input", "backend_result"]
)

# ---------- 10) Agent decision + orchestration ----------
def run_agent_with_backend(query: str) -> Dict[str, Any]:
    """
    High-level orchestration:
    1) Ask LLM whether to call backend (return JSON tool call) or to use SQL (no JSON)
    2) If tool JSON -> execute backend call -> ask LLM to summarize
    3) Otherwise -> run SQL generation + execution + answer generation
    Returns a dict with fields: answer, generated_sql (or None), backend_raw (or None), mode ("backend"|"sql")
    """
    # Build decision prompt to the LLM
    schema_text = get_schema_retriever(query)
    decision_instructions = (
        "Decide whether the user's request requires calling a backend REST API (for actions like create order, update, submit payment, etc.) "
        "or it is purely a data retrieval that should be answered by running SQL against the database. "
        "If you determine a backend call is required, output ONLY a single JSON object matching the BACKEND TOOL format below. "
        "If a backend call is NOT required, output normal text (no JSON tool call)."
        "\n\n" + BACKEND_TOOL_DOC + "\n\n" + BACKEND_TOOL_EXAMPLES
    )

    decision_prompt = f"""
Schema context (trimmed):
{schema_text}

User question: {query}

{decision_instructions}

If returning a tool JSON, ensure it's valid JSON (no trailing text).
"""
    logger.info("Asking LLM to decide action for query.")
    decision_raw = llm.invoke(decision_prompt).content
    logger.debug("Decision raw output: %s", decision_raw[:1000])

    # Try to extract JSON tool call
    tool_obj = extract_first_json(decision_raw)
    if tool_obj and tool_obj.get("tool") == "call_backend":
        logger.info("LLM decided to call backend tool.")
        args = tool_obj.get("arguments", {})
        try:
            status_code, backend_result = execute_backend_call(args)
        except Exception as e:
            logger.error("Backend execution failed: %s", e, exc_info=True)
            # Ask the LLM to explain the backend failure to the user
            failure_text = f"Backend call failed with error: {e}"
            final_answer = answer_llm.invoke(
                backend_answer_prompt.format(input=query, backend_result=failure_text)
            ).content
            return {
                "answer": final_answer,
                "generated_sql": None,
                "backend_raw": {"error": str(e)},
                "mode": "backend"
            }

        # Prepare the backend result to send back to the LLM for summarization
        backend_summary_input = {
            "input": query,
            "backend_result": json.dumps({"status_code": status_code, "body": backend_result}, default=str, indent=2)
        }

        final = answer_llm.invoke(backend_answer_prompt.format(**backend_summary_input)).content

        return {
            "answer": final,
            "generated_sql": None,
            "backend_raw": {"status_code": status_code, "body": backend_result},
            "mode": "backend"
        }

    # --- Fallback path: SQL pipeline ---
    logger.info("LLM did not request backend call; falling back to SQL pipeline.")
    # Generate SQL
    generated_sql = sql_generation_chain.invoke({"input": query})
    # Run SQL (with retry/fix)
    try:
        sql_result = run_sql_query_with_retry(generated_sql, query)
    except Exception as e:
        logger.error("Final SQL execution failed: %s", e, exc_info=True)
        # Let the LLM explain the SQL failure
        fail_resp = answer_llm.invoke(
            f"User question: {query}\nThe system attempted SQL: {generated_sql}\nBut the DB returned an error: {e}\nExplain this to the user and suggest next steps."
        ).content
        return {
            "answer": fail_resp,
            "generated_sql": generated_sql,
            "backend_raw": None,
            "mode": "sql"
        }

    # Use answer prompt to generate final human-friendly answer
    answer_input = {
        "input": query,
        "query": generated_sql,
        "result": sql_result
    }
    # Fill answer_prompt inline to avoid circular import weirdness
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

    final_answer = answer_llm.invoke(answer_prompt.format(**answer_input)).content

    return {
        "answer": final_answer,
        "generated_sql": generated_sql,
        "backend_raw": None,
        "mode": "sql"
    }

# ---------- 11) Public entrypoint ----------
def get_chat_response(query: str) -> Dict[str, Any]:
    """
    Main entrypoint for external code. Returns dict with:
    - answer (str): final NL answer
    - generated_sql (str|None)
    - backend_raw (dict|None)
    - mode ("sql" or "backend")
    """
    return run_agent_with_backend(query)
