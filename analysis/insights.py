from __future__ import annotations

import pandas as pd


def add_case_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an interpretable case label to each turn.

    Case types:
    - error
    - exact
    - semantic_only
    - wrong
    - other
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["case_type"] = "other"

    error_mask = out["error"].notna()
    exact_mask = out["exact_match"] == True
    semantic_only_mask = (out["exact_match"] == False) & (out["execution_match"] == True)
    wrong_mask = out["execution_match"] == False

    out.loc[error_mask, "case_type"] = "error"
    out.loc[~error_mask & exact_mask, "case_type"] = "exact"
    out.loc[~error_mask & semantic_only_mask, "case_type"] = "semantic_only"
    out.loc[~error_mask & wrong_mask, "case_type"] = "wrong"

    return out