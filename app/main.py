from fastapi import FastAPI, HTTPException
from app.models import ChatRequest, ChatResponse
from app.core import get_chat_response
import logging

app = FastAPI(
    title="Postgres RAG Agent",
    description="An API to chat with a PostgreSQL database using RAG on the schema.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/chat", response_model=ChatResponse)
def chat_with_db(request: ChatRequest):
    """
    Receives a natural language query and returns a natural
    language answer and the generated SQL.
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        # Call our core agent logic
        response = get_chat_response(request.query)
        
        logger.info(f"Generated SQL: {response['generated_sql']}")
        return ChatResponse(
            answer=response['answer'],
            generated_sql=response['generated_sql']
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Postgres RAG Agent is running"}