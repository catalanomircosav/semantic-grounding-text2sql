from dataclasses import dataclass
from typing import Any, Optional

from execution import SQLExecutor
from .component_match import component_match


def _normalize_rows(rows, columns=None):
    """
    Convert rows into a canonical comparable format.
    - If rows are dicts, use the provided column order.
    - Ignore column names, preserve column position.
    """
    if not rows:
        return []

    normalized = []

    if isinstance(rows[0], dict):
        if columns is None:
            columns = list(rows[0].keys())

        for row in rows:
            values = [row[col] for col in columns]
            normalized.append(tuple(values))
    else:
        for row in rows:
            normalized.append(tuple(row))

    return normalized


def _execution_match(predicted_result: Optional[dict[str, Any]], gold_result: Optional[dict[str, Any]]) -> bool:
    if predicted_result is None or gold_result is None:
        return False

    pred_rows = _normalize_rows(
        predicted_result.get("rows", []),
        predicted_result.get("columns", []),
    )

    gold_rows = _normalize_rows(
        gold_result.get("rows", []),
        gold_result.get("columns", []),
    )

    try:
        return sorted(pred_rows) == sorted(gold_rows)
    except Exception:
        return pred_rows == gold_rows


def _exact_match(pred_sql: Optional[str], gold_sql: str) -> bool:
    if pred_sql is None:
        return False
    return pred_sql.strip().lower().rstrip(";") == gold_sql.strip().lower().rstrip(";")


@dataclass
class OracleCandidateResult:
    sql: str
    exact_match: bool
    component_match: float
    execution_match: bool
    execution_result: Optional[dict[str, Any]]
    error: Optional[str] = None


@dataclass
class OracleResult:
    oracle_match: bool
    best_sql: Optional[str]
    best_exact_match: bool
    best_component_match: float
    best_execution_match: bool
    best_result: Optional[dict[str, Any]]
    candidate_results: list[OracleCandidateResult]


def evaluate_oracle_at_n(
    candidate_sqls: list[str],
    gold_sql: str,
    db_path: str,
    gold_result: Optional[dict[str, Any]],
) -> OracleResult:
    """
    Evaluate a list of SQL candidates against the gold query/result.

    Selection priority for the best candidate:
    1. execution_match
    2. exact_match
    3. higher component_match

    Returns:
        OracleResult with:
        - oracle_match: whether at least one candidate is execution-correct
        - best_sql: best candidate chosen offline
        - per-candidate results
    """
    executor = SQLExecutor()
    candidate_results: list[OracleCandidateResult] = []

    for sql in candidate_sqls:
        exec_result = executor.execute(db_path=db_path, sql=sql)

        predicted_result = None
        if exec_result.success:
            predicted_result = {
                "columns": exec_result.columns,
                "rows": [list(row) for row in exec_result.rows],
                "row_count": exec_result.row_count,
            }

        exact = _exact_match(sql, gold_sql)
        comp = component_match(sql, gold_sql)
        exec_match = _execution_match(predicted_result, gold_result)

        candidate_results.append(
            OracleCandidateResult(
                sql=sql,
                exact_match=exact,
                component_match=comp,
                execution_match=exec_match,
                execution_result=predicted_result,
                error=None if exec_result.success else exec_result.error,
            )
        )

    if not candidate_results:
        return OracleResult(
            oracle_match=False,
            best_sql=None,
            best_exact_match=False,
            best_component_match=0.0,
            best_execution_match=False,
            best_result=None,
            candidate_results=[],
        )

    # best offline candidate
    best_candidate = sorted(
        candidate_results,
        key=lambda x: (
            x.execution_match,
            x.exact_match,
            x.component_match,
        ),
        reverse=True,
    )[0]

    oracle_match = any(c.execution_match for c in candidate_results)

    return OracleResult(
        oracle_match=oracle_match,
        best_sql=best_candidate.sql,
        best_exact_match=best_candidate.exact_match,
        best_component_match=best_candidate.component_match,
        best_execution_match=best_candidate.execution_match,
        best_result=best_candidate.execution_result,
        candidate_results=candidate_results,
    )