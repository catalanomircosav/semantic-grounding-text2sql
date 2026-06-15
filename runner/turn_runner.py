import json
from typing import Any, Optional

from agents import ConversationState
from data import DatabaseContext, SpiderExample
from execution import SQLExecutor
from experiment_logging import TurnLog
from evaluation import component_match, evaluate_oracle_at_n, candidate_diversity


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


def _extract_predicted_result_table(agent_output: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Attempts to retrieve a result table from the agent output.
    """
    final_answer = agent_output.get("final_answer", "")
    return None if not isinstance(final_answer, dict) else final_answer


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_tool_payload_from_messages(raw_messages: list[Any], tool_name: str) -> Optional[dict[str, Any]]:
    """
    Find the latest ToolMessage with the name "tool_name" and try to parse its JSON.
    """
    for msg in reversed(raw_messages):
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", None)

        if name == tool_name and isinstance(content, str):
            parsed = _safe_json_loads(content)
            if isinstance(parsed, dict):
                return parsed

    return None


def _extract_predicted_sql(raw_messages: list[Any]) -> Optional[str]:
    payload = _extract_tool_payload_from_messages(raw_messages, "execute_sql_query")
    if payload is None:
        return None
    return payload.get("sql_query")


def _extract_predicted_result(raw_messages: list[Any]) -> Optional[dict[str, Any]]:
    payload = _extract_tool_payload_from_messages(raw_messages, "execute_sql_query")
    if payload is None:
        return None
    if payload.get("success") is not True:
        return None
    return {
        "columns": payload.get("columns", []),
        "rows": payload.get("rows", []),
        "row_count": payload.get("row_count", 0),
    }


def _exact_match(pred_sql: Optional[str], gold_sql: str) -> bool:
    if pred_sql is None:
        return False
    return pred_sql.strip().lower().rstrip(";") == gold_sql.strip().lower().rstrip(";")


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

def _serialize_message(msg):
    return {
        "type": type(msg).__name__,
        "name": getattr(msg, "name", None),
        "content": getattr(msg, "content", None),
        "tool_calls": getattr(msg, "tool_calls", None),
    }

def _make_json_safe(obj):
    """
    Recursively convert objects into JSON-serializable structures.
    LangChain message objects are converted via _serialize_message.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]

    if isinstance(obj, tuple):
        return [_make_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # LangChain-style message object
    if hasattr(obj, "content") and hasattr(obj, "__class__"):
        try:
            return _serialize_message(obj)
        except Exception:
            return str(obj)

    return str(obj)

def _execute_sql(executor, db_context, sql: str):
    """
    Execute a SQL query using either:
    - Spider-style executor: execute(db_path=..., sql=...)
    """
    if hasattr(db_context, "database") and not hasattr(db_context, "db_path"):
        return executor.execute(sql)

    return executor.execute(
        db_path=db_context.db_path,
        sql=sql,
    )

class TurnRunner:
    def __init__(self, logger=None, executor=None):
        self.logger = logger
        self.executor = executor or SQLExecutor()

    def run_turn(
        self,
        *,
        run_id: str,
        experiment_name: str,
        agent_type: str,
        model_name: str,
        agent: Any,
        state: ConversationState,
        db_context: DatabaseContext,
        example: SpiderExample,
        turn_id: int,
    ) -> TurnLog:
        # 1. Run agent
        agent_output = agent.run_turn(example.question, state)

        final_answer = agent_output.get("final_answer", "")
        raw_messages = agent_output.get("raw_messages", [])
        turn_messages = agent_output.get("turn_messages", raw_messages)
        error = agent_output.get("error")

        logged_tokens_input = int(agent_output.get("tokens_input", 0) or 0)
        logged_tokens_output = int(agent_output.get("tokens_output", 0) or 0)
        logged_latency_sec = float(agent_output.get("latency_sec", 0.0) or 0.0)

        predicted_sql = None
        predicted_result = None

        candidate_sqls = agent_output.get("candidate_sqls", []) or []
        selection_policy = agent_output.get("selection_policy")
        selected_sql = agent_output.get("selected_sql")

        oracle_result = None
        oracle_match = None
        cand_div = None


        # ---------------------------
        # monolithic baseline
        # ---------------------------
        if candidate_sqls or selected_sql:
            predicted_sql = selected_sql or (candidate_sqls[0] if candidate_sqls else None)
            predicted_result = None

        # ---------------------------
        # Fallback
        # ---------------------------
        else:
            predicted_sql = _extract_predicted_sql(turn_messages)
            predicted_result = _extract_predicted_result(turn_messages)

        # 2. Gold execution
        gold_exec = _execute_sql(
            self.executor,
            db_context,
            example.query,
        )

        gold_result_serializable = None
        if gold_exec.success:
            gold_result_serializable = {
                "columns": gold_exec.columns,
                "rows": [list(row) for row in gold_exec.rows],
                "row_count": gold_exec.row_count,
            }

        # 3. Metrics
        # top-1 metrics
        top1_sql = predicted_sql
        top1_result = predicted_result

        # if baseline and no predicted_result yet, execute selected_sql
        if top1_sql and top1_result is None:
            pred_exec = _execute_sql(
                self.executor,
                db_context,
                top1_sql,
            )
            if pred_exec.success:
                top1_result = {
                    "columns": pred_exec.columns,
                    "rows": [list(row) for row in pred_exec.rows],
                    "row_count": pred_exec.row_count,
                }

        exact_match = _exact_match(top1_sql, example.query)
        execution_match = _execution_match(top1_result, gold_result_serializable)
        comp_match = component_match(top1_sql or "", example.query)

        # baseline oracle@N
        if candidate_sqls:
            cand_div = candidate_diversity(candidate_sqls)

            # Spider / SQLite only for now
            if hasattr(db_context, "db_path"):
                oracle_result = evaluate_oracle_at_n(
                    candidate_sqls=candidate_sqls,
                    gold_sql=example.query,
                    db_path=db_context.db_path,
                    gold_result=gold_result_serializable,
                )
                oracle_match = oracle_result.oracle_match
            else:
                oracle_match = None
        else:
            oracle_match = None
            cand_div = None

        # 4. Build log
        turn_log = TurnLog(
            run_id=run_id,
            experiment_name=experiment_name,
            agent_type=agent_type,
            model_name=model_name,
            db_id=db_context.db_id,
            question_id=turn_id,
            turn_id=turn_id,
            question=example.question,
            model_response=final_answer,
            predicted_sql=top1_sql or "",
            gold_sql=example.query,
            predicted_result=top1_result,
            gold_result=gold_result_serializable,
            exact_match=exact_match,
            execution_match=execution_match,
            component_match=comp_match,
            oracle_match=oracle_match,
            tokens_input=logged_tokens_input,
            tokens_output=logged_tokens_output,
            latency_sec=logged_latency_sec,
            estimated_cost=0.0,
            candidate_sqls=candidate_sqls,
            candidate_diversity=cand_div,
            error=error,
            metadata={
                "raw_message_count": len(raw_messages),
                "turn_message_count": len(turn_messages),
                "serialized_messages": [_serialize_message(m) for m in turn_messages],
                "tokens_input": logged_tokens_input,
                "tokens_output": logged_tokens_output,
                "latency_sec": logged_latency_sec,
                "selection_policy": selection_policy,
                "oracle_best_sql": oracle_result.best_sql if oracle_result is not None else None,
                "oracle_best_exact_match": oracle_result.best_exact_match if oracle_result is not None else None,
                "oracle_best_component_match": oracle_result.best_component_match if oracle_result is not None else None,
                "oracle_best_execution_match": oracle_result.best_execution_match if oracle_result is not None else None,
            },
        )

        # 5. Optional logging
        if self.logger is not None:
            self.logger.log_turn(turn_log)

        return turn_log