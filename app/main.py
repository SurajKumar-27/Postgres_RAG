# app/main.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse  # see suggested models below
from app.core import get_chat_response

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI(
    title="Postgres RAG Agent (Agent + Backend Tools)",
    description="Chat API that can query Postgres or call backend endpoints as an agent.",
    version="1.1.0"
)

# Optional: allow CORS if you use a web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Postgres RAG Agent is running", "version": "1.1.0"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_db(request: ChatRequest):
    """
    Receives a natural language query and returns:
    - answer: final natural language answer
    - generated_sql: the SQL the system executed (if mode == 'sql')
    - backend_raw: raw backend response (if mode == 'backend')
    - mode: 'sql' or 'backend'
    """
    logger.info("Received query: %s", request.query)
    try:
        res = get_chat_response(request.query)
        logger.info("Mode: %s; generated_sql: %s", res.get("mode"), (res.get("generated_sql") or "")[:200])

        # Normalize backend_raw so it's JSON-serializable in response model
        backend_raw = res.get("backend_raw")
        try:
            # ensure it's JSON-serializable; if not, stringify
            import json
            json.dumps(backend_raw)
        except Exception:
            backend_raw = {"_raw": str(backend_raw)}

        return ChatResponse(
            answer=res.get("answer"),
            generated_sql=res.get("generated_sql"),
            backend_raw=backend_raw,
            mode=res.get("mode")
        )
    except Exception as e:
        logger.exception("Error while processing chat request")
        raise HTTPException(status_code=500, detail=str(e))
