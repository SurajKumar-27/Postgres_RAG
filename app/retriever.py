from app.intent import classify_intent
from app.schema_graph import SCHEMA_GRAPH

DOMAIN_TABLES = {
    "PO": ["ezc_po_acknowledgement"],
    "ASN": ["ezc_shipment_header", "ezc_shipment_lines"],
    "GRN": ["ezc_erp_mat_doc_items", "ezc_erp_mat_doc_header"],
    "INVOICE": ["ezc_grn_inv_docs"],
    "USER": ["ezc_users"],
    "CUSTOMER": ["ezc_customer", "ezc_customer_addr"]
}

def hybrid_retrieve(query, indexes):
    # 1) Intent routing
    intents = classify_intent(query)

    allowed_tables = set()
    for i in intents:
        allowed_tables.update(DOMAIN_TABLES.get(i, []))

    # 2) Schema search (filtered)
    if allowed_tables:
        hits = indexes["schema"].similarity_search(
            query,
            k=10,
            filter={"table": {"$in": list(allowed_tables)}}
        )
    else:
        hits = indexes["schema"].similarity_search(query, k=10)

    # 3) Table voting
    votes = {}
    for h in hits:
        t = h.metadata["table"]
        votes[t] = votes.get(t, 0) + 1

    top_tables = sorted(votes, key=votes.get, reverse=True)[:3]

    return {
        "tables": top_tables,
        "columns": hits
    }
