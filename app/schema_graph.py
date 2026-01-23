SCHEMA_GRAPH = {
    "ezc_po_acknowledgement": {
        "pk": "ezpa_doc_no",
        "joins": {
            "ezc_shipment_header": "ezpa_doc_no = ezsh_po_num"
        }
    },
    "ezc_shipment_header": {
        "pk": "ezsh_sh_id",
        "joins": {
            "ezc_shipment_lines": "ezsh_sh_id = ezsl_sh_id",
            "ezc_erp_mat_doc_items": "ezsh_sh_id = ship_id"
        }
    },
    "ezc_erp_mat_doc_items": {
        "pk": "mat_doc",
        "joins": {
            "ezc_grn_inv_docs": "mat_doc = egid_grn_no"
        }
    }
}
