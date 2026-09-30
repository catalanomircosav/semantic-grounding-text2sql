import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import ClarificationAgent, ConversationState
from data import SpiderDataset
from execution import SQLExecutor
from models.openai_chat_model import get_openai_chat_model
from .db_setup import prepare_database


@dataclass
class ClarificationExperimentResult:
    run_id: str
    model_name: str
    semantic_enrichment: bool
    num_scenarios: int
    initial_action_accuracy: float
    resolved_action_accuracy: Optional[float]
    resolved_execution_accuracy: Optional[float]
    end_to_end_accuracy: float
    direct_resolved_action_accuracy: float
    direct_resolved_execution_accuracy: float
    action_discrimination_accuracy: float
    log_path: str

def _execution_match(pred_rows, gold_rows) -> bool:
    try:
        return sorted(pred_rows) == sorted(gold_rows)
    except Exception:
        return list(pred_rows) == list(gold_rows)


class ClarificationExperimentRunner:
    """
    Evaluate the clarification-aware monolithic agent on controlled
    two-turn ambiguity scenarios.

    Each scenario is treated as an independent conversation.
    """

    def __init__(
        self,
        *,
        spider_path: str | Path = "data/spider",
        evaluation_set_path: str | Path = (
            "evaluation_sets/clarification_concert_singer.json"
        ),
        outputs_dir: str | Path = "outputs/clarification",
    ):
        self.spider_path = Path(spider_path)
        self.evaluation_set_path = Path(evaluation_set_path)
        self.outputs_dir = Path(outputs_dir)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_scenarios(self) -> list[dict[str, Any]]:
        with self.evaluation_set_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            scenarios = json.load(f)

        if not isinstance(scenarios, list):
            raise ValueError("Clarification evaluation set must be a JSON list")

        return scenarios

    def run(
        self,
        *,
        run_id: str,
        model_name: str,
        temperature: float = 0.0,
        memory_window: int = 10,
        semantic_enrichment: bool = False,
        verbose: bool = False,
        overwrite: bool = False,
    ) -> ClarificationExperimentResult:

        scenarios = self._load_scenarios()

        dataset = SpiderDataset(self.spider_path)
        db_id = "concert_singer"

        schema_json = dataset.build_schema_database_json(db_id)
        db_path = dataset.resolve_db_path(db_id)

        package = prepare_database(
            db_id=db_id,
            spider_path=self.spider_path,
            eval_limit=0,
        )

        model = get_openai_chat_model(
            model_name=model_name,
            temperature=temperature,
        )

        executor = SQLExecutor()

        safe_model_name = model_name.replace("/", "_")
        grounding_label = (
            "semantic"
            if semantic_enrichment
            else "no_semantic"
        )

        log_path = self.outputs_dir / (
            f"{run_id}__{db_id}__{safe_model_name}"
            f"__{grounding_label}.jsonl"
        )

        if log_path.exists() and not overwrite:
            raise FileExistsError(
                f"Log already exists: {log_path}. "
                "Use overwrite=True only if you intentionally want to replace it."
            )

        initial_correct = 0
        resolved_action_correct = 0
        resolved_execution_correct = 0
        end_to_end_correct = 0

        direct_resolved_action_correct = 0
        direct_resolved_execution_correct = 0

        num_resolved_turns = 0

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:

            for scenario in scenarios:

                # Each scenario is an independent two-turn conversation.
                state = ConversationState(db_id=db_id)

                agent = ClarificationAgent(
                    model=model,
                    schema_json=schema_json,
                    fewshot_retriever=package.fewshot_retriever,
                    memory_window=memory_window,
                    verbose=verbose,
                    db_id=db_id,
                    semantic_enrichment=semantic_enrichment,
                )

                first = agent.run_turn(
                    scenario["ambiguous_question"],
                    state,
                )

                expected_initial_action = scenario[
                    "expected_action_ambiguous"
                ]

                initial_action_match = (
                    first.get("action")
                    == expected_initial_action
                )

                if initial_action_match:
                    initial_correct += 1

                second = None
                resolved_action_match = False
                resolved_execution_match = False

                predicted_rows = None
                gold_rows = None
                predicted_execution_error = None
                gold_execution_error = None

                # Continue the conversation only when the model actually
                # asks for clarification.
                if first.get("action") == "CLARIFY":
                    num_resolved_turns += 1

                    second = agent.run_turn(
                        scenario["user_resolution"],
                        state,
                    )

                    resolved_action_match = (
                        second.get("action")
                        == scenario["expected_action_resolved"]
                    )

                    if resolved_action_match:
                        resolved_action_correct += 1

                    predicted_sql = second.get("sql")

                    if (
                        second.get("action") == "ANSWER"
                        and predicted_sql
                    ):
                        pred_exec = executor.execute(
                            db_path=db_path,
                            sql=predicted_sql,
                        )

                        gold_exec = executor.execute(
                            db_path=db_path,
                            sql=scenario["gold_sql"],
                        )

                        predicted_execution_error = pred_exec.error
                        gold_execution_error = gold_exec.error

                        if pred_exec.success:
                            predicted_rows = [
                                list(row)
                                for row in pred_exec.rows
                            ]

                        if gold_exec.success:
                            gold_rows = [
                                list(row)
                                for row in gold_exec.rows
                            ]

                        if pred_exec.success and gold_exec.success:
                            resolved_execution_match = _execution_match(
                                pred_exec.rows,
                                gold_exec.rows,
                            )

                    if resolved_execution_match:
                        resolved_execution_correct += 1

                direct_state = ConversationState(db_id=db_id)

                direct = agent.run_turn(
                    scenario["resolved_question"],
                    direct_state,
                )

                direct_action_match = (
                    direct.get("action")
                    == scenario["expected_action_resolved"]
                )

                if direct_action_match:
                    direct_resolved_action_correct += 1

                direct_execution_match = False

                direct_predicted_rows = None
                direct_gold_rows = None

                direct_predicted_execution_error = None
                direct_gold_execution_error = None

                direct_sql = direct.get("sql")

                direct_gold_exec = executor.execute(
                    db_path=db_path,
                    sql=scenario["gold_sql"],
                )

                direct_gold_execution_error = direct_gold_exec.error

                if direct_gold_exec.success:
                    direct_gold_rows = [
                        list(row)
                        for row in direct_gold_exec.rows
                    ]

                if (
                    direct.get("action") == "ANSWER"
                    and direct_sql
                ):
                    direct_pred_exec = executor.execute(
                        db_path=db_path,
                        sql=direct_sql,
                    )

                    direct_predicted_execution_error = (
                        direct_pred_exec.error
                    )

                    if direct_pred_exec.success:
                        direct_predicted_rows = [
                            list(row)
                            for row in direct_pred_exec.rows
                        ]

                    if (
                        direct_pred_exec.success
                        and direct_gold_exec.success
                    ):
                        direct_execution_match = _execution_match(
                            direct_pred_exec.rows,
                            direct_gold_exec.rows,
                        )

                if direct_execution_match:
                    direct_resolved_execution_correct += 1
                
                end_to_end_success = (
                    initial_action_match
                    and resolved_action_match
                    and resolved_execution_match
                )

                if end_to_end_success:
                    end_to_end_correct += 1

                first_tokens_input = int(
                    first.get("tokens_input", 0) or 0
                )
                first_tokens_output = int(
                    first.get("tokens_output", 0) or 0
                )
                first_latency = float(
                    first.get("latency_sec", 0.0) or 0.0
                )

                second_tokens_input = 0
                second_tokens_output = 0
                second_latency = 0.0

                if second is not None:
                    second_tokens_input = int(
                        second.get("tokens_input", 0) or 0
                    )
                    second_tokens_output = int(
                        second.get("tokens_output", 0) or 0
                    )
                    second_latency = float(
                        second.get("latency_sec", 0.0) or 0.0
                    )

                direct_tokens_input = int(
                    direct.get("tokens_input", 0) or 0
                )

                direct_tokens_output = int(
                    direct.get("tokens_output", 0) or 0
                )

                direct_latency = float(
                    direct.get("latency_sec", 0.0) or 0.0
                )
                record = {
                    "run_id": run_id,
                    "model_name": model_name,
                    "db_id": db_id,
                    "semantic_enrichment": semantic_enrichment,

                    "scenario_id": scenario["id"],
                    "ambiguity_type": scenario["ambiguity_type"],

                    "ambiguous_question": scenario[
                        "ambiguous_question"
                    ],
                    "expected_action_ambiguous": (
                        expected_initial_action
                    ),

                    "predicted_initial_action": first.get("action"),
                    "predicted_clarification": first.get(
                        "clarification"
                    ),
                    "initial_sql": first.get("sql"),
                    "initial_action_match": initial_action_match,
                    "initial_error": first.get("error"),

                    "clarification_target": scenario[
                        "clarification_target"
                    ],
                    "user_resolution": scenario["user_resolution"],
                    "resolved_question": scenario["resolved_question"],
                    "expected_action_resolved": scenario[
                        "expected_action_resolved"
                    ],

                    "predicted_resolved_action": (
                        second.get("action")
                        if second is not None
                        else None
                    ),
                    "predicted_resolved_sql": (
                        second.get("sql")
                        if second is not None
                        else None
                    ),
                    "resolved_action_match": resolved_action_match,
                    "resolved_execution_match": (
                        resolved_execution_match
                    ),
                    
                    "direct_resolved_question": scenario[
                        "resolved_question"
                    ],
                    "direct_expected_action": scenario[
                        "expected_action_resolved"
                    ],
                    "direct_predicted_action": direct.get("action"),
                    "direct_predicted_clarification": direct.get(
                        "clarification"
                    ),
                    "direct_predicted_sql": direct.get("sql"),
                    "direct_action_match": direct_action_match,
                    "direct_execution_match": direct_execution_match,
                    "direct_predicted_rows": direct_predicted_rows,
                    "direct_gold_rows": direct_gold_rows,
                    "direct_predicted_execution_error": (
                        direct_predicted_execution_error
                    ),
                    "direct_gold_execution_error": (
                        direct_gold_execution_error
                    ),
                    "direct_error": direct.get("error"),

                    "gold_sql": scenario["gold_sql"],
                    "predicted_rows": predicted_rows,
                    "gold_rows": gold_rows,

                    "predicted_execution_error": (
                        predicted_execution_error
                    ),
                    "gold_execution_error": gold_execution_error,

                    "end_to_end_success": end_to_end_success,

                    "tokens_input_first": first_tokens_input,
                    "tokens_output_first": first_tokens_output,
                    "latency_sec_first": first_latency,

                    "tokens_input_second": second_tokens_input,
                    "tokens_output_second": second_tokens_output,
                    "latency_sec_second": second_latency,

                    "tokens_input_direct": direct_tokens_input,
                    
                    "tokens_output_direct": direct_tokens_output,
                    "latency_sec_direct": direct_latency,

                    "tokens_input_experiment_total": (
                        first_tokens_input
                        + second_tokens_input
                        + direct_tokens_input
                    ),
                    "tokens_output_experiment_total": (
                        first_tokens_output
                        + second_tokens_output
                        + direct_tokens_output
                    ),
                    "latency_sec_experiment_total": (
                        first_latency
                        + second_latency
                        + direct_latency
                    ),

                    "direct_conversation_history": list(
                        direct_state.history
                    ),

                    "tokens_input_total": (
                        first_tokens_input
                        + second_tokens_input
                    ),
                    "tokens_output_total": (
                        first_tokens_output
                        + second_tokens_output
                    ),
                    "latency_sec_total": (
                        first_latency
                        + second_latency
                    ),

                    "conversation_history": list(state.history),
                }

                log_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        total = len(scenarios)

        initial_accuracy = (
            initial_correct / total
            if total
            else 0.0
        )

        resolved_action_accuracy = (
            resolved_action_correct / num_resolved_turns
            if num_resolved_turns
            else None
        )

        resolved_execution_accuracy = (
            resolved_execution_correct / num_resolved_turns
            if num_resolved_turns
            else None
        )

        end_to_end_accuracy = (
            end_to_end_correct / total
            if total
            else 0.0
        )

        direct_resolved_action_accuracy = (
            direct_resolved_action_correct / total
            if total
            else 0.0
        )

        direct_resolved_execution_accuracy = (
            direct_resolved_execution_correct / total
            if total
            else 0.0
        )

        action_discrimination_accuracy = (
            (
                initial_correct
                + direct_resolved_action_correct
            )
            / (2 * total)
            if total
            else 0.0
        )
        
        return ClarificationExperimentResult(
            run_id=run_id,
            model_name=model_name,
            semantic_enrichment=semantic_enrichment,
            num_scenarios=total,
            initial_action_accuracy=initial_accuracy,
            resolved_action_accuracy=resolved_action_accuracy,
            resolved_execution_accuracy=resolved_execution_accuracy,
            end_to_end_accuracy=end_to_end_accuracy,
            direct_resolved_action_accuracy=(
                direct_resolved_action_accuracy
            ),
            direct_resolved_execution_accuracy=(
                direct_resolved_execution_accuracy
            ),
            action_discrimination_accuracy=(
                action_discrimination_accuracy
            ),
            log_path=str(log_path),
        )