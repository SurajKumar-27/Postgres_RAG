from __future__ import annotations

from io import BytesIO
from typing import Tuple

import pandas as pd
from pypdf import PdfReader


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
