def candidate_diversity(candidate_sqls: list[str]) -> float:
    """
    Diversity = number of unique candidates / total candidates.
    """
    if not candidate_sqls:
        return 0.0

    normalized = [sql.strip() for sql in candidate_sqls if isinstance(sql, str) and sql.strip()]
    if not normalized:
        return 0.0

    return len(set(normalized)) / len(normalized)