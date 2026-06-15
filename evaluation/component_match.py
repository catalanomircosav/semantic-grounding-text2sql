import re
from typing import Dict


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().lower())


def _extract_component(sql: str, keyword: str, next_keywords: list[str]) -> str:
    """
    Extracts the contents of an SQL clause (SELECT, FROM, WHERE, etc.)
    """
    sql = _normalize_sql(sql)

    pattern = rf"{keyword}\s+(.*?)\s+(?={'|'.join(next_keywords)})"
    match = re.search(pattern, sql)

    if match:
        return match.group(1).strip()

    # fallback: until the end
    pattern = rf"{keyword}\s+(.*)"
    match = re.search(pattern, sql)

    if match:
        return match.group(1).strip()

    return ""


def _decompose_sql(sql: str) -> Dict[str, str]:
    return {
        "select": _extract_component(sql, "select", ["from"]),
        "from": _extract_component(sql, "from", ["where", "group by", "order by", "limit"]),
        "where": _extract_component(sql, "where", ["group by", "order by", "limit"]),
        "group_by": _extract_component(sql, "group by", ["order by", "limit"]),
        "order_by": _extract_component(sql, "order by", ["limit"]),
    }


def _component_equal(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def component_match(pred_sql: str, gold_sql: str) -> float:
    """
    Returns a score between 0 and 1 based on matching SQL components.
    """
    if not pred_sql:
        return 0.0

    pred = _decompose_sql(pred_sql)
    gold = _decompose_sql(gold_sql)

    total = len(pred)
    matches = 0

    for key in pred:
        if _component_equal(pred[key], gold[key]):
            matches += 1

    return matches / total if total > 0 else 0.0