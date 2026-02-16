from __future__ import annotations

from io import BytesIO
from typing import Tuple

import pandas as pd
from pypdf import PdfReader

import os
import requests
import logging
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

def execute_remote_query(sql_query: str) -> list:
    """Executes T-SQL via the remote MSSQL REST endpoint with authentication."""
    url = os.getenv("REMOTE_DB_URL")
    user = os.getenv("REMOTE_DB_USER")
    password = os.getenv("REMOTE_DB_PASSWORD")
    
    if not url:
        raise ValueError("REMOTE_DB_URL is not set in the environment.")

    payload = {"query": sql_query}
    
    try:
        # Using HTTPBasicAuth based on your requirement for user/password headers
        response = requests.post(
            url, 
            json=payload, 
            auth=HTTPBasicAuth(user, password),
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        # Standardize response: return the recordset list
        if isinstance(data, list):
            return data
        return data.get("recordset") or data.get("results") or []
        
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
