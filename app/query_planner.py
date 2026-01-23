from app.schema_graph import SCHEMA_GRAPH

def build_query_plan(tables):
    joins = []
    for t in tables:
        rules = SCHEMA_GRAPH.get(t, {}).get("joins", {})
        for target, rule in rules.items():
            if target in tables:
                joins.append(rule)

    return {
        "tables": tables,
        "joins": joins
    }
