def validate_sql(sql: str, allowed_tables: list[str]) -> bool:
    sql_lower = sql.lower()
    for t in allowed_tables:
        if t.lower() in sql_lower:
            return True
    return False
