from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .aggregate import load_logs_to_dataframe
from .insights import add_case_type

from evaluation import relational_accuracy


def _ensure_path_list(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(p) for p in paths]


def load_many_logs(log_paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Load multiple JSONL experiment logs into a single DataFrame.

    Adds:
    - log_path
    - log_file
    """
    paths = _ensure_path_list(log_paths)

    frames = []
    for path in paths:
        if not path.exists():
            continue

        df = load_logs_to_dataframe(path)
        if df.empty:
            continue

        df = df.copy()
        df["log_path"] = str(path)
        df["log_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


def discover_logs(
    logs_dir: str | Path,
    pattern: str = "*.jsonl",
) -> list[Path]:
    """
    Discover log files in a directory.
    """
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return []

    return sorted(logs_dir.glob(pattern))


def build_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per log/run.
    Useful to inspect each produced JSONL separately.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = add_case_type(df).copy()

    tmp["exact_match"] = tmp["exact_match"].fillna(False).astype(float)
    tmp["execution_match"] = tmp["execution_match"].fillna(False).astype(float)
    tmp["oracle_match"] = tmp["oracle_match"].fillna(False).astype(float)
    tmp["component_match"] = pd.to_numeric(tmp["component_match"], errors="coerce").fillna(0.0)
    tmp["candidate_diversity"] = pd.to_numeric(tmp["candidate_diversity"], errors="coerce").fillna(0.0)
    tmp["tokens_input"] = pd.to_numeric(tmp["tokens_input"], errors="coerce").fillna(0.0)
    tmp["tokens_output"] = pd.to_numeric(tmp["tokens_output"], errors="coerce").fillna(0.0)
    tmp["latency_sec"] = pd.to_numeric(tmp["latency_sec"], errors="coerce").fillna(0.0)

    group_cols = [
        "log_file",
        "experiment_name",
        "agent_type",
        "model_name",
        "db_id",
    ]
    group_cols = [c for c in group_cols if c in tmp.columns]

    out = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            num_turns=("turn_id", "count"),
            exact_ratio=("exact_match", "mean"),
            execution_ratio=("execution_match", "mean"),
            oracle_ratio=("oracle_match", "mean"),
            component_match_mean=("component_match", "mean"),
            candidate_diversity_mean=("candidate_diversity", "mean"),
            avg_tokens_input=("tokens_input", "mean"),
            avg_tokens_output=("tokens_output", "mean"),
            avg_total_tokens=(("tokens_input"), "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            semantic_only_ratio=("case_type", lambda s: (s == "semantic_only").mean()),
            wrong_ratio=("case_type", lambda s: (s == "wrong").mean()),
            error_ratio=("case_type", lambda s: (s == "error").mean()),
        )
        .reset_index()
    )

    # fix avg_total_tokens = avg input + avg output
    if not out.empty:
        out["avg_total_tokens"] = out["avg_tokens_input"] + out["avg_tokens_output"]

    return out.sort_values(["db_id", "experiment_name"]).reset_index(drop=True)


def build_setup_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all DBs for each setup/experiment.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = add_case_type(df).copy()

    numeric_cols = [
        "exact_match",
        "execution_match",
        "oracle_match",
        "component_match",
        "candidate_diversity",
        "tokens_input",
        "tokens_output",
        "latency_sec",
    ]
    for col in numeric_cols:
        if col in ["exact_match", "execution_match", "oracle_match"]:
            tmp[col] = tmp[col].fillna(False).astype(float)
        else:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0.0)

    group_cols = ["experiment_name", "agent_type", "model_name"]
    group_cols = [c for c in group_cols if c in tmp.columns]

    out = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            num_turns=("turn_id", "count"),
            num_dbs=("db_id", "nunique"),
            exact_ratio=("exact_match", "mean"),
            execution_ratio=("execution_match", "mean"),
            oracle_ratio=("oracle_match", "mean"),
            component_match_mean=("component_match", "mean"),
            candidate_diversity_mean=("candidate_diversity", "mean"),
            relational_accuracy_mean=("relational_accuracy", "mean"),
            informational_relational_accuracy_mean=("informational_relational_accuracy", "mean"),
            avg_tokens_input=("tokens_input", "mean"),
            avg_tokens_output=("tokens_output", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            semantic_only_ratio=("case_type", lambda s: (s == "semantic_only").mean()),
            wrong_ratio=("case_type", lambda s: (s == "wrong").mean()),
            error_ratio=("case_type", lambda s: (s == "error").mean()),
        )
        .reset_index()
    )

    if not out.empty:
        out["avg_total_tokens"] = out["avg_tokens_input"] + out["avg_tokens_output"]

    return out.sort_values(["execution_ratio", "component_match_mean"], ascending=False).reset_index(drop=True)


def build_db_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all setups for each DB.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = add_case_type(df).copy()

    numeric_cols = [
        "exact_match",
        "execution_match",
        "oracle_match",
        "component_match",
        "candidate_diversity",
        "tokens_input",
        "tokens_output",
        "latency_sec",
    ]
    for col in numeric_cols:
        if col in ["exact_match", "execution_match", "oracle_match"]:
            tmp[col] = tmp[col].fillna(False).astype(float)
        else:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0.0)

    out = (
        tmp.groupby(["db_id"], dropna=False)
        .agg(
            num_turns=("turn_id", "count"),
            num_setups=("experiment_name", "nunique"),
            exact_ratio=("exact_match", "mean"),
            execution_ratio=("execution_match", "mean"),
            oracle_ratio=("oracle_match", "mean"),
            component_match_mean=("component_match", "mean"),
            candidate_diversity_mean=("candidate_diversity", "mean"),
            avg_tokens_input=("tokens_input", "mean"),
            avg_tokens_output=("tokens_output", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            semantic_only_ratio=("case_type", lambda s: (s == "semantic_only").mean()),
            wrong_ratio=("case_type", lambda s: (s == "wrong").mean()),
            error_ratio=("case_type", lambda s: (s == "error").mean()),
        )
        .reset_index()
    )

    if not out.empty:
        out["avg_total_tokens"] = out["avg_tokens_input"] + out["avg_tokens_output"]

    return out.sort_values("db_id").reset_index(drop=True)


def build_db_setup_matrix(
    df: pd.DataFrame,
    value_col: str = "execution_match",
    index_col: str = "db_id",
    columns_col: str = "experiment_name",
) -> pd.DataFrame:
    """
    Pivot table to compare setups across DBs.

    Example:
    - rows = db_id
    - cols = experiment_name
    - values = mean execution_match
    """
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()

    if value_col in ["exact_match", "execution_match", "oracle_match"]:
        tmp[value_col] = tmp[value_col].fillna(False).astype(float)
    else:
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0)

    pivot = pd.pivot_table(
        tmp,
        index=index_col,
        columns=columns_col,
        values=value_col,
        aggfunc="mean",
    )

    return pivot.sort_index()


def build_global_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-row global summary across all loaded runs.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = add_case_type(df).copy()

    tmp["exact_match"] = tmp["exact_match"].fillna(False).astype(float)
    tmp["execution_match"] = tmp["execution_match"].fillna(False).astype(float)
    tmp["oracle_match"] = tmp["oracle_match"].fillna(False).astype(float)
    tmp["component_match"] = pd.to_numeric(tmp["component_match"], errors="coerce").fillna(0.0)
    tmp["candidate_diversity"] = pd.to_numeric(tmp["candidate_diversity"], errors="coerce").fillna(0.0)
    tmp["tokens_input"] = pd.to_numeric(tmp["tokens_input"], errors="coerce").fillna(0.0)
    tmp["tokens_output"] = pd.to_numeric(tmp["tokens_output"], errors="coerce").fillna(0.0)
    tmp["latency_sec"] = pd.to_numeric(tmp["latency_sec"], errors="coerce").fillna(0.0)

    out = pd.DataFrame([{
        "num_logs": tmp["log_file"].nunique() if "log_file" in tmp.columns else 0,
        "num_dbs": tmp["db_id"].nunique() if "db_id" in tmp.columns else 0,
        "num_setups": tmp["experiment_name"].nunique() if "experiment_name" in tmp.columns else 0,
        "num_turns": len(tmp),
        "exact_ratio": tmp["exact_match"].mean(),
        "execution_ratio": tmp["execution_match"].mean(),
        "oracle_ratio": tmp["oracle_match"].mean(),
        "component_match_mean": tmp["component_match"].mean(),
        "candidate_diversity_mean": tmp["candidate_diversity"].mean(),
        "avg_tokens_input": tmp["tokens_input"].mean(),
        "avg_tokens_output": tmp["tokens_output"].mean(),
        "avg_total_tokens": tmp["tokens_input"].mean() + tmp["tokens_output"].mean(),
        "avg_latency_sec": tmp["latency_sec"].mean(),
        "semantic_only_ratio": (tmp["case_type"] == "semantic_only").mean(),
        "wrong_ratio": (tmp["case_type"] == "wrong").mean(),
        "error_ratio": (tmp["case_type"] == "error").mean(),
    }])

    return out


def filter_runs(
    df: pd.DataFrame,
    *,
    experiment_names: Optional[list[str]] = None,
    db_ids: Optional[list[str]] = None,
    model_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convenient filtering helper.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if experiment_names is not None and "experiment_name" in out.columns:
        out = out[out["experiment_name"].isin(experiment_names)]

    if db_ids is not None and "db_id" in out.columns:
        out = out[out["db_id"].isin(db_ids)]

    if model_names is not None and "model_name" in out.columns:
        out = out[out["model_name"].isin(model_names)]

    return out.reset_index(drop=True)

def add_advanced_metrics(
    df: pd.DataFrame,
    *,
    alpha: float = 2.0,
    beta: float = 1.0,
):
    out = df.copy()

    rel_scores = []

    for _, row in out.iterrows():
        pred = row.get("predicted_result")
        gold = row.get("gold_result")

        rel = relational_accuracy(
            predicted_result=pred,
            gold_result=gold,
            alpha=alpha,
            beta=beta,
        )

        rel_scores.append(rel)

    out["relational_accuracy"] = rel_scores

    return out