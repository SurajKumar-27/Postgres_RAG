from langchain_google_vertexai import ChatVertexAI
import json

llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)

DOMAINS = {
    "PO": ["purchase order", "po", "vendor", "plant", "pur org"],
    "ASN": ["asn", "shipment", "ibd", "inbound", "delivery"],
    "GRN": ["gr", "goods receipt", "grn", "mat doc"],
    "INVOICE": ["invoice", "ocr", "egid", "billing"],
    "USER": ["created by", "user", "processed by"],
    "CUSTOMER": ["customer", "business partner"]
}

def classify_intent(query: str) -> list[str]:
    prompt = f"""
Classify this query into business domains:
{json.dumps(DOMAINS, indent=2)}

Return a JSON array only.
Query: {query}
"""
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return []
