from __future__ import annotations

from io import BytesIO
from typing import List, Tuple, Any

import pandas as pd
from pypdf import PdfReader

import os
import requests
import logging

logger = logging.getLogger(__name__)


def execute_remote_query(sql_query: str) -> List[Any]:
    """Executes T-SQL via the remote MSSQL REST endpoint with custom headers."""
    url = os.getenv("REMOTE_DB_URL")
    user_val = os.getenv("REMOTE_DB_USER")
    pass_val = os.getenv("REMOTE_DB_PASSWORD")
    
    headers = {
        "user": user_val,
        "password": pass_val,
        "Content-Type": "application/json"
    }
    payload = {"query": sql_query}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result_json = response.json()
        
        # 🆕 Handle the specific structure: {"data": [...]}
        if isinstance(result_json, dict) and "data" in result_json:
            return result_json["data"]
            
        # Fallback for other potential formats
        if isinstance(result_json, list):
            return result_json
            
        return result_json.get("recordset") or result_json.get("results") or []
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Remote MSSQL query failed: {e}")
        raise Exception(f"Database API Connectivity Error: {str(e)}")


def extract_text_from_pdf(content: bytes) -> Tuple[str, int]:
    reader = PdfReader(BytesIO(content))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    text = "\n".join(pages_text).strip()
    return text, len(reader.pages)


def extract_text_from_excel(content: bytes) -> Tuple[str, int]:
    sheets = pd.read_excel(BytesIO(content), sheet_name=None)
    parts = []
    for sheet_name, dataframe in sheets.items():
        parts.append(f"Sheet: {sheet_name}")
        parts.append(dataframe.to_csv(index=False))
    text = "\n".join(parts).strip()
    return text, len(sheets)
