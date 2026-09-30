import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import ConversationalExecutionFeedbackAgent, ConversationState
from data import SpiderDataset
from evaluation import (
    component_match,
    quadratic_mean,
    relational_accuracy,
    symbiosis_score,
)
from evaluation.advanced_metrics import informational_relational_accuracy
from execution import SQLExecutor
from experiment_logging import JSONLLogger, TurnLog
from models.openai_chat_model import get_openai_chat_model

from .db_setup import prepare_database

SYMBIOSIS_CAPS = {"tokens": 10000.0, "latency": 5.0}
SYMBIOSIS_MU = 2.0


@dataclass
class ConversationalExecutionFeedbackResult:
    run_id: str
    db_id: str
    num_examples: int
    metrics: dict[str, float]
    log_path: str


def _load_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    records = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if source.suffix.lower() == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(records, list):
        raise TypeError("Evaluation set must contain a JSON list or JSONL records")
    return records


def _table(result) -> dict[str, Any] | None:
    if not result.success:
        return None
    return {
        "columns": result.columns,
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
    }


def _rows_match(left, right) -> bool:
    try:
        return sorted(left) == sorted(right)
    except TypeError:
        return list(left) == list(right)


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records) if records else 0.0


class ConversationalExecutionFeedbackRunner:
    """Evaluate Spider and ambiguous questions through the same agent flow."""

    def __init__(
        self,
        *,
        spider_path: str | Path = "data/spider",
        outputs_dir: str | Path = "outputs/conversational_execution_feedback",
        faq_dir: str | Path = "enrichment/faqs",
    ):
        self.spider_path = Path(spider_path)
        self.outputs_dir = Path(outputs_dir)
        self.faq_dir = Path(faq_dir)

    def run(
        self,
        *,
        run_id: str,
        db_id: str,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        semantic_enrichment: bool = False,
        use_faq: bool = False,
        faq_threshold: float = 0.45,
        faq_top_k: int = 1,
        revision_policy: str = "always",
        evaluation_set_path: str | Path | None = None,
        include_spider: bool = True,
        limit: int | None = None,
        memory_window: int = 10,
        feedback_row_limit: int = 20,
        symbiosis_caps: dict[str, float] | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> ConversationalExecutionFeedbackResult:
        package = prepare_database(
            db_id=db_id,
            spider_path=self.spider_path,
            eval_limit=None,
        )
        dataset = SpiderDataset(self.spider_path)
        examples = []
        if include_spider:
            examples.extend(
                {
                    "id": f"spider_{index}",
                    "source": "spider",
                    "question": example.question,
                    "gold_sql": example.query,
                    "expected_action": "ANSWER",
                }
                for index, example in enumerate(package.eval_turns)
            )
        if evaluation_set_path:
            examples.extend(self._ambiguous_examples(evaluation_set_path))
        if limit is not None:
            examples = examples[:limit]

        model = get_openai_chat_model(model_name=model_name, temperature=temperature)
        schema_json = dataset.build_schema_database_json(db_id)
        executor = SQLExecutor()
        caps = dict(symbiosis_caps or SYMBIOSIS_CAPS)

        def new_agent() -> ConversationalExecutionFeedbackAgent:
            return ConversationalExecutionFeedbackAgent(
                model=model,
                schema_json=schema_json,
                db_path=str(package.context.db_path),
                fewshot_retriever=package.fewshot_retriever,
                memory_window=memory_window,
                verbose=verbose,
                db_id=db_id,
                semantic_enrichment=semantic_enrichment,
                feedback_row_limit=feedback_row_limit,
                use_faq=use_faq,
                faq_dir=self.faq_dir,
                faq_threshold=faq_threshold,
                faq_top_k=faq_top_k,
                revision_policy=revision_policy,
            )

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model_name.replace("/", "_")
        log_path = self.outputs_dir / f"{run_id}__{db_id}__{safe_model}.jsonl"
        if log_path.exists() and not overwrite:
            raise FileExistsError(f"Log already exists: {log_path}")

        records = []
        with JSONLLogger(log_path) as logger:
            for index, example in enumerate(examples):
                record, turn_log = self._run_example(
                    example=example,
                    index=index,
                    run_id=run_id,
                    db_id=db_id,
                    model_name=model_name,
                    temperature=temperature,
                    semantic_enrichment=semantic_enrichment,
                    use_faq=use_faq,
                    revision_policy=revision_policy,
                    agent=new_agent(),
                    db_path=package.context.db_path,
                    executor=executor,
                    caps=caps,
                )
                records.append(record)
                logger.log_turn(turn_log)

        return ConversationalExecutionFeedbackResult(
            run_id=run_id,
            db_id=db_id,
            num_examples=len(records),
            metrics=self._metrics(records),
            log_path=str(log_path),
        )

    @staticmethod
    def _ambiguous_examples(path: str | Path) -> list[dict[str, Any]]:
        examples = []
        for index, item in enumerate(_load_json_records(path)):
            examples.append(
                {
                    "id": item.get("id", f"ambiguous_{index}"),
                    "source": "ambiguous",
                    "question": item.get("ambiguous_question", item.get("question")),
                    "gold_sql": item["gold_sql"],
                    "user_resolution": item.get("user_resolution"),
                    "expected_action": item.get("expected_action_ambiguous", "CLARIFY"),
                }
            )
        return examples

    @staticmethod
    def _run_example(
        *,
        example,
        index,
        run_id,
        db_id,
        model_name,
        temperature,
        semantic_enrichment,
        use_faq,
        revision_policy,
        agent,
        db_path,
        executor,
        caps,
    ):
        state = ConversationState(db_id=db_id)
        first = agent.run_turn(example["question"], state)
        final = first
        if first.get("action") == "CLARIFY" and example.get("user_resolution"):
            final = agent.run_turn(example["user_resolution"], state)

        gold = executor.execute(db_path, example["gold_sql"])
        initial = executor.execute(db_path, final.get("initial_sql"))
        predicted = executor.execute(db_path, final.get("selected_sql"))
        gold_table = _table(gold)
        predicted_table = _table(predicted)
        execution_match = bool(
            predicted.success and gold.success and _rows_match(predicted.rows, gold.rows)
        )
        initial_match = bool(
            initial.success and gold.success and _rows_match(initial.rows, gold.rows)
        )
        exact_match = bool(
            final.get("selected_sql")
            and final["selected_sql"].strip().lower().rstrip(";")
            == example["gold_sql"].strip().lower().rstrip(";")
        )
        component = component_match(final.get("selected_sql") or "", example["gold_sql"])
        relational = relational_accuracy(predicted_table, gold_table)
        informational = informational_relational_accuracy(predicted_table, gold_table)
        tokens_input = int(first.get("tokens_input", 0)) + (
            int(final.get("tokens_input", 0)) if final is not first else 0
        )
        tokens_output = int(first.get("tokens_output", 0)) + (
            int(final.get("tokens_output", 0)) if final is not first else 0
        )
        latency = float(first.get("latency_sec", 0.0)) + (
            float(final.get("latency_sec", 0.0)) if final is not first else 0.0
        )
        tool_calls = int(first.get("num_tool_calls", 0)) + (
            int(final.get("num_tool_calls", 0)) if final is not first else 0
        )
        llm_calls = int(first.get("num_llm_calls", 0)) + (
            int(final.get("num_llm_calls", 0)) if final is not first else 0
        )
        revision_performed = bool(
            first.get("revision_performed") or final.get("revision_performed")
        )
        revision_trigger = (
            final.get("revision_trigger")
            if final is not first
            else first.get("revision_trigger")
        )
        symbiosis_result = symbiosis_score(
            accuracy=relational,
            resources={"tokens": tokens_input + tokens_output, "latency": latency},
            caps=caps,
            resource_agg_fn=quadratic_mean,
            mu=SYMBIOSIS_MU,
        )
        action_match = first.get("action") == example["expected_action"]
        resolved_action_match = bool(
            final is not first and final.get("action") == "ANSWER"
        )
        record = {
            "source": example["source"],
            "initial_action": first.get("action"),
            "action_match": action_match,
            "resolved": final is not first,
            "resolved_action": final.get("action") if final is not first else None,
            "resolved_action_match": resolved_action_match,
            "initial_execution_correct": initial_match,
            "execution_match": execution_match,
            "exact_match": exact_match,
            "component_match": component,
            "relational_accuracy": relational,
            "informational_relational_accuracy": informational,
            "symbiosis": symbiosis_result["symbiosis"],
            "symbiosis_resource_score": symbiosis_result["resource_score"],
            "symbiosis_decay": symbiosis_result["decay"],
            "query_changed": bool(
                final.get("initial_sql")
                and final.get("selected_sql")
                and final["initial_sql"] != final["selected_sql"]
            ),
            "helpful_revision": bool(not initial_match and execution_match),
            "harmful_revision": bool(initial_match and not execution_match),
            "end_to_end_success": bool(
                action_match
                and execution_match
                and (example["source"] == "spider" or resolved_action_match)
            ),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency_sec": latency,
            "num_tool_calls": tool_calls,
            "num_llm_calls": llm_calls,
            "revision_performed": revision_performed,
        }
        metadata = {
            **record,
            "example_id": example["id"],
            "temperature": temperature,
            "semantic_enrichment": semantic_enrichment,
            "use_faq": use_faq,
            "revision_policy": revision_policy,
            "revision_trigger": revision_trigger,
            "clarification": first.get("clarification"),
            "user_resolution": example.get("user_resolution"),
            "initial_sql": final.get("initial_sql"),
            "execution_feedback": final.get("execution_feedback"),
            "faq_id": final.get("faq_id"),
        }
        turn_log = TurnLog(
            run_id=run_id,
            experiment_name="conversational_execution_feedback",
            agent_type="conversational_execution_feedback",
            model_name=model_name,
            db_id=db_id,
            question_id=index,
            turn_id=index,
            question=example["question"],
            model_response=final.get("final_answer", ""),
            predicted_sql=final.get("selected_sql") or "",
            gold_sql=example["gold_sql"],
            predicted_result=predicted_table,
            gold_result=gold_table,
            exact_match=exact_match,
            execution_match=execution_match,
            component_match=component,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_sec=latency,
            num_tool_calls=tool_calls,
            candidate_sqls=final.get("candidate_sqls", []),
            error=final.get("error"),
            metadata=metadata,
        )
        return record, turn_log

    @staticmethod
    def _metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        ambiguous = [record for record in records if record["source"] == "ambiguous"]
        spider = [record for record in records if record["source"] == "spider"]
        resolved = [record for record in ambiguous if record["resolved"]]
        return {
            "initial_execution_accuracy": _mean(records, "initial_execution_correct"),
            "final_execution_accuracy": _mean(records, "execution_match"),
            "exact_match": _mean(records, "exact_match"),
            "component_match": _mean(records, "component_match"),
            "relational_accuracy": _mean(records, "relational_accuracy"),
            "informational_relational_accuracy": _mean(
                records, "informational_relational_accuracy"
            ),
            "symbiosis": _mean(records, "symbiosis"),
            "avg_symbiosis_resource_score": _mean(
                records, "symbiosis_resource_score"
            ),
            "avg_symbiosis_decay": _mean(records, "symbiosis_decay"),
            "action_accuracy": _mean(records, "action_match"),
            "spider_answer_rate": _mean(spider, "action_match"),
            "spider_execution_accuracy": _mean(spider, "execution_match"),
            "ambiguity_recall": _mean(ambiguous, "action_match"),
            "resolved_action_accuracy": _mean(resolved, "resolved_action_match"),
            "resolved_execution_accuracy": _mean(resolved, "execution_match"),
            "end_to_end_accuracy": _mean(records, "end_to_end_success"),
            "ambiguity_end_to_end_accuracy": _mean(
                ambiguous, "end_to_end_success"
            ),
            "query_changed_rate": _mean(records, "query_changed"),
            "helpful_revision_rate": _mean(records, "helpful_revision"),
            "harmful_revision_rate": _mean(records, "harmful_revision"),
            "avg_tokens_input": _mean(records, "tokens_input"),
            "avg_tokens_output": _mean(records, "tokens_output"),
            "avg_tokens": _mean(records, "tokens_input")
            + _mean(records, "tokens_output"),
            "avg_latency_sec": _mean(records, "latency_sec"),
            "avg_tool_calls": _mean(records, "num_tool_calls"),
            "revision_call_rate": _mean(records, "revision_performed"),
            "avg_llm_calls": _mean(records, "num_llm_calls"),
            "total_tokens_input": sum(record["tokens_input"] for record in records),
            "total_tokens_output": sum(record["tokens_output"] for record in records),
            "total_latency_sec": sum(record["latency_sec"] for record in records),
            "total_tool_calls": sum(record["num_tool_calls"] for record in records),
            "total_llm_calls": sum(record["num_llm_calls"] for record in records),
        }
