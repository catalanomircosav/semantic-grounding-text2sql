from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import (
    BaselineAgent,
    ConversationState,
)
from agents.execution_feedback_agent import ExecutionFeedbackAgent
from data import SpiderDataset
from experiment_logging import JSONLLogger
from models.openai_chat_model import get_openai_chat_model

from .db_setup import prepare_database
from .turn_runner import TurnRunner


@dataclass
class ExperimentResult:
    run_id: str
    experiment_name: str
    db_id: str
    num_turns: int
    num_completed_turns: int
    log_path: str


class _StaticAgentOutputAdapter:
    """
    Small adapter so TurnRunner can consume a precomputed agent output
    with the same .run_turn(...) interface.
    """

    def __init__(self, output: dict[str, Any]):
        self.output = output

    def run_turn(self, question: str, state: ConversationState) -> dict[str, Any]:
        return self.output


class ExperimentRunner:
    """
    Runs a full conversational Text-to-SQL experiment on a single database.

    This version supports:
    - automatic DB preparation
    - log writing
    - simple resume/skip based on existing JSONL logs
    """

    def __init__(
        self,
        spider_path: str | Path = "../data/spider",
        outputs_log_dir: str | Path = "../outputs/logs",
    ):
        self.spider_path = Path(spider_path)
        self.outputs_log_dir = Path(outputs_log_dir)
        self.outputs_log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_log_path(
        outputs_log_dir: Path,
        experiment_name: str,
        db_id: str,
        model_name: str,
    ) -> Path:
        safe_model_name = model_name.replace("/", "_")
        filename = f"{experiment_name}__{db_id}__{safe_model_name}.jsonl"
        return outputs_log_dir / filename


    def run_baseline_experiment(
        self,
        *,
        run_id: str,
        experiment_name: str,
        db_id: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        eval_start: int = 0,
        eval_limit: int | None = 50,
        memory_window: int = 50,
        n_samples: int = 5,
        max_workers: int | None = None,
        force_catalog_regeneration: bool = False,
        verbose: bool = False,
        log_path: str | Path | None = None,
        resume: bool = True,
        semantic_enrichment: bool = False,
    ) -> ExperimentResult:
        # 1. Prepare DB package
        package = prepare_database(
            db_id=db_id,
            spider_path=self.spider_path,
            model_name=model_name,
            eval_start=eval_start,
            eval_limit=eval_limit,
        )

        # 2. Full schema for monolithic baseline
        dataset = SpiderDataset(self.spider_path)
        schema_json = dataset.build_schema_database_json(db_id)

        # 3. Build model
        llm = get_openai_chat_model(
            model_name=model_name,
            temperature=temperature,
        )

        # 4. Build baseline agent
        baseline_agent = BaselineAgent(
            model=llm,
            schema_json=schema_json,
            fewshot_retriever=package.fewshot_retriever,
            memory_window=memory_window,
            verbose=verbose,
            db_id=db_id,
            semantic_enrichment=semantic_enrichment
        )

        # 5. State
        state = ConversationState(db_id=package.context.db_id)

        # 6. Logger / resume
        if log_path is None:
            log_path = self._default_log_path(
                self.outputs_log_dir,
                experiment_name=experiment_name,
                db_id=db_id,
                model_name=model_name,
            )

        logger = JSONLLogger(log_path)
        turn_runner = TurnRunner(logger=logger)

        existing_keys = set()
        if resume:
            existing_keys = JSONLLogger.existing_keys(
                log_path,
                key_fields=("run_id", "db_id", "turn_id"),
            )

        completed = 0

        try:
            for turn_id, example in enumerate(package.eval_turns, start=eval_start):
                key = (run_id, db_id, turn_id)

                if resume and key in existing_keys:
                    continue

                # sample N candidates in parallel
                sample_output = baseline_agent.sample_candidates_parallel(
                    question=example.question,
                    state=state,
                    n_samples=n_samples,
                    max_workers=max_workers,
                )

                # update conversational memory with top-1 selected sql
                selected_sql = sample_output.get("selected_sql")
                state.history.append({"role": "user", "content": example.question})
                state.history.append(
                    {
                        "role": "assistant",
                        "content": selected_sql or "",
                    }
                )
                state.turn_index += 1

                # pass a baseline-shaped output to TurnRunner
                baseline_output = {
                    "question": example.question,
                    "final_answer": selected_sql or "",
                    "selected_sql": selected_sql,
                    "candidate_sqls": sample_output.get("candidate_sqls", []),
                    "selection_policy": sample_output.get("selection_policy", "parallel_sampling"),
                    "raw_messages": sample_output.get("raw_messages", []),
                    "raw_texts": sample_output.get("raw_texts", []),
                    "tokens_input": sample_output.get("tokens_input", 0),
                    "tokens_output": sample_output.get("tokens_output", 0),
                    "latency_sec": sample_output.get("latency_sec", 0.0),
                    "error": sample_output.get("error"),
                }

                _ = turn_runner.run_turn(
                    run_id=run_id,
                    experiment_name=experiment_name,
                    agent_type="baseline_monolithic_semantic" if semantic_enrichment else "baseline_monolithic",
                    model_name=model_name,
                    agent=_StaticAgentOutputAdapter(baseline_output),
                    state=state,
                    db_context=package.context,
                    example=example,
                    turn_id=turn_id,
                )
                completed += 1

        finally:
            logger.close()

        return ExperimentResult(
            run_id=run_id,
            experiment_name=experiment_name,
            db_id=db_id,
            num_turns=len(package.eval_turns),
            num_completed_turns=completed,
            log_path=str(log_path),
        )
        
        
    def run_execution_feedback_experiment(
        self,
        *,
        run_id: str,
        experiment_name: str,
        db_id: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        eval_start: int = 0,
        eval_limit: int | None = 50,
        memory_window: int = 50,
        semantic_enrichment: bool = False,
        feedback_row_limit: int = 20,
        verbose: bool = False,
        log_path: str | Path | None = None,
        resume: bool = True,
    ) -> ExperimentResult:

        package = prepare_database(
            db_id=db_id,
            spider_path=self.spider_path,
            model_name=model_name,
            eval_start=eval_start,
            eval_limit=eval_limit,
        )

        dataset = SpiderDataset(self.spider_path)
        schema_json = dataset.build_schema_database_json(db_id)

        llm = get_openai_chat_model(
            model_name=model_name,
            temperature=temperature,
        )

        agent = ExecutionFeedbackAgent(
            model=llm,
            schema_json=schema_json,
            db_path=str(package.context.db_path),
            fewshot_retriever=package.fewshot_retriever,
            memory_window=memory_window,
            verbose=verbose,
            db_id=db_id,
            semantic_enrichment=semantic_enrichment,
            feedback_row_limit=feedback_row_limit,
        )

        state = ConversationState(db_id=package.context.db_id)

        if log_path is None:
            log_path = self._default_log_path(
                self.outputs_log_dir,
                experiment_name=experiment_name,
                db_id=db_id,
                model_name=model_name,
            )

        logger = JSONLLogger(log_path)
        turn_runner = TurnRunner(logger=logger)

        existing_keys = set()
        if resume:
            existing_keys = JSONLLogger.existing_keys(
                log_path,
                key_fields=("run_id", "db_id", "turn_id"),
            )

        completed = 0

        try:
            for turn_id, example in enumerate(package.eval_turns, start=eval_start):
                key = (run_id, db_id, turn_id)

                if resume and key in existing_keys:
                    continue

                agent_type = (
                    "baseline_monolithic_semantic_execution_feedback"
                    if semantic_enrichment
                    else "baseline_monolithic_execution_feedback"
                )

                _ = turn_runner.run_turn(
                    run_id=run_id,
                    experiment_name=experiment_name,
                    agent_type=agent_type,
                    model_name=model_name,
                    agent=agent,
                    state=state,
                    db_context=package.context,
                    example=example,
                    turn_id=turn_id,
                )

                completed += 1

        finally:
            logger.close()

        return ExperimentResult(
            run_id=run_id,
            experiment_name=experiment_name,
            db_id=db_id,
            num_turns=len(package.eval_turns),
            num_completed_turns=completed,
            log_path=str(log_path),
        )