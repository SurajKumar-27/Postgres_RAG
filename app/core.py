# app/core.py
import ast
import os
import logging
import json
import re
from typing import List
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import text


import requests
# ... other imports ...

from app.intent import classify_intent
from app.indexes import load_indexes
from app.query_planner import build_query_plan
from app.sql_guard import validate_sql


# 1. Change these imports
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


import requests
from langchain_core.output_parsers import CommaSeparatedListOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import sessionmaker
from numpy import dot
from numpy.linalg import norm
import numpy as np

from app.database import db_engine, SchemaEmbedding

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Environment / Config ----------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
# Optional: token to include in backend requests
BACKEND_AUTH_TOKEN = os.getenv("BACKEND_AUTH_TOKEN", None)
INDEXES = load_indexes()

# ---------- 1) LLMs & Embeddings ----------
# Core LLM used for SQL generation / decisioning. Keep temp = 0 for deterministic SQL.
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001", 
    temperature=0, 
    project=PROJECT_ID, 
    location=LOCATION
)
# Answer generation LLM (you can pick a faster model)
answer_llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001", 
    temperature=0, 
    project=PROJECT_ID, 
    location=LOCATION
)
# Embedding model (schema retrieval)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004", # Production-ready embedding model
    project=PROJECT_ID,
    location=LOCATION
)
# ---------- 2) Database setup ----------
db = SQLDatabase(db_engine)
VectorSession = sessionmaker(bind=db_engine)

# rag/app/core.py

# ... (keep existing imports)

def get_schema_retriever(query: str, k: int = 40) -> str:
    session = VectorSession()
    try:
        # ---------- 1) Intent ----------
        intents = classify_intent(query)

        allowed_tables = set()
        from app.retriever import DOMAIN_TABLES
        for i in intents:
            allowed_tables.update(DOMAIN_TABLES.get(i, []))

        # ---------- 2) Embedding ----------
        query_embedding = embeddings.embed_query(query)
        schemas = session.query(SchemaEmbedding).all()

        # ---------- 3) Similarity + Filter ----------
        scored = []
        for s in schemas:
            if allowed_tables and s.table_name not in allowed_tables:
                continue

            emb = np.array(s.embedding)
            sim = dot(emb, query_embedding) / (norm(emb) * norm(query_embedding) + 1e-8)
            scored.append((sim, s))

        top_k = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        if not top_k:
            return "NO RELEVANT SCHEMA FOUND"

        # ---------- 4) Table Voting ----------
        votes = {}
        for _, s in top_k:
            votes[s.table_name] = votes.get(s.table_name, 0) + 1

        top_tables = sorted(votes, key=votes.get, reverse=True)[:3]

        # ---------- 5) Query Plan (Join Rules) ----------
        plan = build_query_plan(top_tables)

        # ---------- 6) Load DDL ----------
        ddl_text = db.get_table_info(table_names=top_tables)

        # ---------- 7) Best Columns ----------
        cols = []
        rules = []
        seen = set()
        for _, s in top_k:
            desc = s.description.strip()

            if "BUSINESS_RULE" in desc:
                rules.append(desc)
                continue

            if s.column_name and s.column_name not in seen:
                cols.append(desc)
                seen.add(s.column_name)

            if len(cols) >= 8 and len(rules) >= 5:
                break

        return f"""
### ALLOWED TABLES:
{", ".join(top_tables)}

### JOIN RULES:
{chr(10).join(plan['joins'])}

### TABLE SCHEMAS:
{ddl_text}

### RELEVANT COLUMNS:
{chr(10).join("- " + c for c in cols)}

### BUSINESS RULES (MUST FOLLOW):
{chr(10).join("- " + r for r in rules)}
"""

    finally:
        session.close()


# app/core.py

# suggestion_prompt = PromptTemplate(
#     template="""
# You are a helpful data assistant with access to a PostgreSQL database.
# User Question: {input}
# Current Answer: {answer}

# Relevant Database Schema:
# {schema}

# Based on the answer and the available schema, suggest 3 short, relevant follow-up questions the user might want to ask next to explore this data further. 
# Ensure the questions can actually be answered using the provided schema.
# Return ONLY the questions, separated by commas. Do not number them.

# Example: "Show sales breakdown by region, Who are the top 5 customers, List late deliveries"
# """,
#     input_variables=["input", "answer", "schema"]
# )

# app/core.py

suggestion_prompt = PromptTemplate(
    template="""
You are a helpful data assistant.
User Question: {input}
Current Answer: {answer}

Based on the answer and available schema, suggest 3 short, relevant follow-up questions.

STRICT RULES:
1. Use ONLY natural, business-friendly language.
2. NEVER include technical column names, table names, or SQL snippets in the suggestions.
3. Ensure the questions are answerable using the provided schema.
4. Return ONLY the questions, separated by commas. Do not number them.

Example: "What is the total revenue by month?, Who are the top performing agents?, Show recent high-value orders"

Relevant Database Schema:
{schema}
""",
    input_variables=["input", "answer", "schema"]
)

# app/core.py

def generate_suggestions(query: str, answer: str, schema: str) -> List[str]:
    """Generates 3 follow-up questions based on context and schema."""
    try:
        # Pass the schema context to the LLM
        prompt_text = suggestion_prompt.format(
            input=query, 
            answer=answer, 
            schema=schema
        )
        response = answer_llm.invoke(prompt_text).content
        
        # Simple parsing to get a list
        questions = [q.strip() for q in response.split(",") if q.strip()]
        return questions[:3] 
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return []
    

contextualize_prompt = PromptTemplate(
    template="""
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

User Question: {input}

Standalone Question:
""",
    input_variables=["chat_history", "input"]
)

def contextualize_query(query: str, history: List[Any]) -> str:
    """Rewrites the query to be standalone based on history."""
    if not history:
        return query
    
    # Format history into a string: "User: hi\nAssistant: hello"
    history_str = "\n".join([f"{msg.role.title()}: {msg.content}" for msg in history])
    
    try:
        # Use the fast LLM (answer_llm) for this text manipulation
        new_query = answer_llm.invoke(
            contextualize_prompt.format(chat_history=history_str, input=query)
        ).content
        logger.info(f"Contextualized Query: '{query}' -> '{new_query}'")
        return new_query
    except Exception as e:
        logger.error(f"Contextualization failed: {e}")
        return query

# ---------- 4) SQL generation chain ----------
# sql_prompt = PromptTemplate(
#     template="""
# You are an expert PostgreSQL analyst. Given a user question and a schema description,
# write a single, syntactically correct SELECT SQL query that answers the question,
# or return an appropriate read-only query (no UPDATE/DELETE/INSERT/DROP).
# Only use tables and columns from the provided schema.

# User Question: {input}
# Relevant Schema:
# {schema}

# SQL Query:
# """,
#     input_variables=["input", "schema"]
# )
# app/core.py
# sql_prompt = PromptTemplate(
#     template="""
# You are an ERP SQL engine.

# ALLOWED TABLES & JOIN RULES:
# {schema}

# Rules:
# - Use ONLY allowed tables
# - Follow JOIN RULES exactly
# - Prefer LEFT JOIN
# - Always include LIMIT 200
# - If the answer cannot be found, return: INSUFFICIENT DATA

# User Question:
# {input}

# SQL:
# """,
#     input_variables=["input", "schema"]
# )
sql_prompt = PromptTemplate(
    template="""
You are an ERP SQL engine.

KEYWORD MAPPING (PRIMARY TABLES):
- **"ASN" / "Shipment"** -> Start with `FROM ezc_shipment_header`
- **"PO" / "Purchase Order"** -> Start with `FROM ezc_po_acknowledgement`
- **"GRN" / "Goods Receipt"** -> Start with `FROM ezc_erp_mat_doc_items`
- **"Invoice"** -> Start with `FROM ezc_grn_inv_docs`

ALLOWED TABLES & RULES:
{schema}

STRICT GUIDELINES:
1. **Identify the Primary Entity**: Look at the user's request. If they ask for "ASN", you MUST use `ezc_shipment_header`. Do NOT query `ezc_po_acknowledgement` for ASNs.
2. **JOIN for Context**: If the user asks for details not in the primary table (e.g., "ASN with Vendor Name" or "PO with Invoice Amount"), you MUST `JOIN` the relevant tables.
3. **Missing Columns**: Always JOIN `ezc_users` for Creator Names and `ezc_customer` for Vendor Names.
4. **Limit**: Default to `LIMIT 200` unless the user asks for a specific number.

User Question:
{input}

SQL:
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
        schema_ctx = get_schema_retriever(question)
        match = re.search(r"ALLOWED TABLES:\n(.+)", schema_ctx)
        if match:
            allowed = [t.strip() for t in match.group(1).split(",")]
            if not validate_sql(cleaned, allowed):
                raise Exception("SQL blocked: references tables outside allowed scope")

        with db_engine.connect() as conn:
            result = conn.execute(text(cleaned))
            return result.mappings().all()
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
        with db_engine.connect() as conn:
            result = conn.execute(text(cleaned_fixed))
            return result.mappings().all()


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
# Example 1 (order details):
# {
#  "tool": "call_backend",
#  "arguments": {
#    "method": "POST",
#    "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_regular_order_details_guest",
#    "headers": {"Content-Type": "application/json"},
#    "body": {"salesOrderNo": "...."}
#  }
# }

# Example 2 (get invoice list):
# {
#  "tool": "call_backend",
#  "arguments": {
#    "method": "POST",
#    "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_invoice_list",
#    "headers": {"Content-Type": "application/json"},
#    "body": {"soldTo":"...", "fromDate":"...", "toDate": "..."}
#  }
# }

# Example 3 (invoice details):
# {
#  "tool": "call_backend",
#  "arguments": {
#    "method": "POST",
#    "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_invoice_details",
#    "headers": {"Content-Type": "application/json"},
#    "body": {"invoiceNo": "...."}
#  }
# }

# Example 4 (delivery details):
# {
#  "tool": "call_backend",
#  "arguments": {
#    "method": "POST",
#    "url": "https://ifonmbbhyreuewdcvfyt.supabase.co/functions/v1/sls_delivery_details",
#    "headers": {"Content-Type": "application/json"},
#    "body": {"deliveryNo": "...."}
#  }
# }
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
def run_agent_with_backend(query: str, history: List[Any] = []) -> Dict[str, Any]:
    
    # 🆕 STEP 1: Contextualize the query
    # If the user says "and by region?", we convert it to "Show sales by region"
    effective_query = contextualize_query(query, history)
    """
    High-level orchestration:
    1) Ask LLM whether to call backend (return JSON tool call) or to use SQL (no JSON)
    2) If tool JSON -> execute backend call -> ask LLM to summarize
    3) Otherwise -> run SQL generation + execution + answer generation
    Returns a dict with fields: answer, generated_sql (or None), backend_raw (or None), mode ("backend"|"sql")
    """
    # Build decision prompt to the LLM
    schema_text = get_schema_retriever(effective_query)
    decision_instructions = (
        "Decide whether the user's request requires calling a backend REST API (for actions like create order, update, submit payment, etc.) "
        "or it is purely a data retrieval that should be answered by running SQL against the database. "
        "If you determine a backend call is required, output ONLY a single JSON object matching the BACKEND TOOL format below. "
        "If a backend call is NOT required, output normal text (no JSON tool call)."
        # "\n\n" + BACKEND_TOOL_DOC + "\n\n" + BACKEND_TOOL_EXAMPLES
    )

    decision_prompt = f"""
Schema context (trimmed):
{schema_text}

User question: {effective_query}

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
    # ... inside run_agent_with_backend ...

    # Generate SQL
    generated_sql = sql_generation_chain.invoke({"input": effective_query})

    try:
        result_list = run_sql_query_with_retry(generated_sql, effective_query)

        # try:
        #     # db.run returns a string like "[(1, 'a'), (2, 'b')]". We convert it to a python list.
        #     if not raw_result:
        #         result_list = []
        #     else:
        #         result_list = ast.literal_eval(raw_result)
        # except Exception as e:
        #     logger.error(f"Failed to parse SQL result string: {e}")
        #     result_list = []
        
        # FIX 1: Remove json.dumps from this helper so it returns a Dict
        def format_sql_result(result, max_rows=200):
            if not result:
                return {"row_count": 0, "sample_rows": []}

            formatted = []
            for row in result[:max_rows]:
                # 1. Try explicit conversion to dict (works for SQLAlchemy RowMapping)
                try:
                    formatted.append(dict(row))
                except (ValueError, TypeError):
                    # 2. Fallback for tuple-like results if dict conversion fails
                    formatted.append({
                        f"col_{i+1}": value
                        for i, value in enumerate(row)
                    })
            
            return {
                "row_count": len(result),
                "sample_rows": formatted
            }

        # sql_payload is now a Dict
        sql_payload = format_sql_result(result_list)

        # FIX 2: Now this check works because sql_payload is a dictionary
        if sql_payload["row_count"] == 0:
            return {
                "answer": "I couldn’t find any records that match your request. Try adjusting your filters, date range, or search criteria.",
                "suggested_questions": generate_suggestions(
                    query,
                    "No records found",
                    schema_text
                ),
                "generated_sql": generated_sql,
                "backend_raw": None,
                "mode": "sql"
            }

        # FIX 3: Convert to JSON string here, just before sending to LLM
        sql_result = json.dumps(sql_payload, indent=2, default=str)

    except Exception as e:
        # ... (rest of your exception handling)
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
#     answer_prompt = PromptTemplate(
#         template="""
# Given a user question, the SQL query you generated, and the result of that query,
# formulate a concise, natural language answer.

# User Question: {input}
# SQL Query: {query}
# SQL Result: {result}

# Answer:
# """,
#         input_variables=["input", "query", "result"]
#     )

# app/core.py inside run_agent_with_backend()

    answer_prompt = PromptTemplate(
        template="""
Given a user question, the SQL query used, and the raw database results, 
formulate a concise, natural language answer.

STRICT GUIDELINES:
1. Do NOT mention internal database column names or table names.
2. Use human-friendly business labels.
3. If the user asks for a specific number of items (e.g., "100 users", "50 orders"), you MUST list ALL of them.
4. Do NOT summarize with phrases like "and X others" or "including others".
5. Present long lists clearly, preferably using a markdown list or comma separation.


User Question: {input}
SQL Query: {query}
SQL Result: {result}

Answer:
""",
        input_variables=["input", "query", "result"]
    )

    # final_answer = answer_llm.invoke(answer_prompt.format(**answer_input)).content
    final_response = answer_llm.invoke(answer_prompt.format(**answer_input))
    final_answer = final_response.content
    suggestions = generate_suggestions(query, final_answer, schema_text)

    if final_response.usage_metadata:
        # Returns dict like: {'input_tokens': 150, 'output_tokens': 45, 'total_tokens': 195}
        logger.info(f"Final Answer Usage: {final_response.usage_metadata}")

    return {
        "answer": final_answer,
        "suggested_questions": suggestions, # 🆕 Add this
        "generated_sql": generated_sql,
        "backend_raw": None,
        "mode": "sql"
    }

# ---------- 11) Public entrypoint ----------
def get_chat_response(query: str, history: List[Any] = []) -> Dict[str, Any]:
    return run_agent_with_backend(query, history)
