from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .experiment_runner import ExperimentRunner, ExperimentResult


@dataclass
class ExperimentSpec:
    setup_name: str
    agent_type: str
    model_name: str
    temperature: float = 0.0
    memory_window: int = 50
    eval_limit: Optional[int] = None
    n_samples: int = 5
    max_workers_internal: Optional[int] = None


@dataclass
class GridRunItemResult:
    db_id: str
    setup_name: str
    success: bool
    log_path: Optional[str]
    error: Optional[str] = None


@dataclass
class GridRunResult:
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    items: list[GridRunItemResult]


class GridRunner:
    """
    Run many experiments across:
    - multiple DBs
    - multiple setups
    """

    def __init__(
        self,
        spider_path: str | Path = "../data/spider",
        outputs_log_dir: str | Path = "../outputs/logs",
    ):
        self.exp_runner = ExperimentRunner(
            spider_path=spider_path,
            outputs_log_dir=outputs_log_dir,
        )

    def _run_one(
        self,
        *,
        db_id: str,
        spec: ExperimentSpec,
        resume: bool = True,
    ) -> GridRunItemResult:
        run_id = f"{spec.setup_name}__{db_id}"
        experiment_name = spec.setup_name

        try:

            if spec.agent_type == "baseline":
                result = self.exp_runner.run_baseline_experiment(
                    run_id=run_id,
                    experiment_name=experiment_name,
                    db_id=db_id,
                    model_name=spec.model_name,
                    temperature=spec.temperature,
                    eval_limit=spec.eval_limit,
                    memory_window=spec.memory_window,
                    n_samples=spec.n_samples,
                    max_workers=spec.max_workers_internal,
                    verbose=False,
                    resume=resume,
                )

            else:
                return GridRunItemResult(
                    db_id=db_id,
                    setup_name=spec.setup_name,
                    success=False,
                    log_path=None,
                    error=f"Unknown agent_type: {spec.agent_type}",
                )

            return GridRunItemResult(
                db_id=db_id,
                setup_name=spec.setup_name,
                success=True,
                log_path=result.log_path,
                error=None,
            )

        except Exception as e:
            return GridRunItemResult(
                db_id=db_id,
                setup_name=spec.setup_name,
                success=False,
                log_path=None,
                error=str(e),
            )

    def run_grid(
        self,
        *,
        db_ids: list[str],
        specs: list[ExperimentSpec],
        max_workers: int = 2,
        resume: bool = True,
    ) -> GridRunResult:
        jobs = []
        for db_id in db_ids:
            for spec in specs:
                jobs.append((db_id, spec))

        results: list[GridRunItemResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._run_one,
                    db_id=db_id,
                    spec=spec,
                    resume=resume,
                )
                for db_id, spec in jobs
            ]

            for fut in as_completed(futures):
                results.append(fut.result())

        completed = sum(1 for x in results if x.success)
        failed = sum(1 for x in results if not x.success)

        return GridRunResult(
            total_jobs=len(jobs),
            completed_jobs=completed,
            failed_jobs=failed,
            items=results,
        )