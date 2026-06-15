from pathlib import Path
from typing import Optional

import pandas as pd

from experiment_logging import JSONLLogger


def load_logs_to_dataframe(log_path: str | Path) -> pd.DataFrame:
    """
    Load a JSONL experiment log into a pandas DataFrame.
    """
    records = JSONLLogger.read_all(log_path)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure expected columns exist even if missing in some rows
    expected_columns = [
        "run_id",
        "experiment_name",
        "agent_type",
        "model_name",
        "db_id",
        "question_id",
        "turn_id",
        "question",
        "model_response",
        "predicted_sql",
        "gold_sql",
        "predicted_result",
        "gold_result",
        "exact_match",
        "component_match",
        "execution_match",
        "oracle_match",
        "tokens_input",
        "tokens_output",
        "latency_sec",
        "estimated_cost",
        "candidate_sqls",
        "error",
        "metadata",
        "timestamp",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    return df


def build_turn_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a readable per-turn analysis table.
    """
    if df.empty:
        return pd.DataFrame()

    turn_df = df.copy()

    columns = [
        "turn_id",
        "question",
        "predicted_sql",
        "gold_sql",
        "exact_match",
        "component_match",
        "execution_match",
        #"num_tool_calls",
        #"num_tables_used",
        #"retrieved_tables_str",
        "error",
    ]

    existing_columns = [c for c in columns if c in turn_df.columns]
    turn_df = turn_df[existing_columns].sort_values("turn_id").reset_index(drop=True)

    return turn_df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row summary table for the experiment.
    """
    if df.empty:
        return pd.DataFrame()

    summary = {
        "num_turns": len(df),
        "exact_match_mean": df["exact_match"].fillna(False).astype(float).mean(),
        "component_match_mean": pd.to_numeric(df["component_match"], errors="coerce").fillna(0.0).mean(),
        "execution_match_mean": df["execution_match"].fillna(False).astype(float).mean(),
        "num_errors": df["error"].notna().sum(),
        "avg_latency_sec": pd.to_numeric(df["latency_sec"], errors="coerce").fillna(0.0).mean(),
        "avg_estimated_cost": pd.to_numeric(df["estimated_cost"], errors="coerce").fillna(0.0).mean(),
        "avg_tokens_input": pd.to_numeric(df["tokens_input"], errors="coerce").fillna(0.0).mean(),
        "avg_tokens_output": pd.to_numeric(df["tokens_output"], errors="coerce").fillna(0.0).mean(),
    }

    summary_df = pd.DataFrame([summary])
    return summary_df


def build_error_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show only turns with errors.
    """
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()

    err_df = df[df["error"].notna()].copy()
    if err_df.empty:
        return pd.DataFrame()

    columns = [
        "turn_id",
        "question",
        "predicted_sql",
        "error",
    ]
    existing_columns = [c for c in columns if c in err_df.columns]
    return err_df[existing_columns].sort_values("turn_id").reset_index(drop=True)


def build_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show turns where at least one main metric is not fully correct.
    Useful for quick debugging.
    """
    if df.empty:
        return pd.DataFrame()

    mismatch_df = df[
        (df["exact_match"] != True) | (df["execution_match"] != True)
    ].copy()

    if mismatch_df.empty:
        return pd.DataFrame()

    columns = [
        "turn_id",
        "question",
        "predicted_sql",
        "gold_sql",
        "exact_match",
        "component_match",
        "execution_match",
        "error",
    ]

    existing_columns = [c for c in columns if c in mismatch_df.columns]
    return mismatch_df[existing_columns].sort_values("turn_id").reset_index(drop=True)

def build_query_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact table focused only on SQL generation quality.
    Useful for prompt and query-writing inspection.
    """
    if df.empty:
        return pd.DataFrame()

    query_df = df.copy()

    columns = [
        "turn_id",
        "question",
        "predicted_sql",
        "gold_sql",
        "exact_match",
        "component_match",
        "execution_match",
        "error",
    ]

    existing_columns = [c for c in columns if c in query_df.columns]
    query_df = query_df[existing_columns].sort_values("turn_id").reset_index(drop=True)

    return query_df