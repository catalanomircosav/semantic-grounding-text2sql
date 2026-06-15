from __future__ import annotations

import itertools
import math
from typing import Any, Optional


SQLResult = Optional[dict[str, Any]]
Relation = set[tuple[Any, ...]]


def _normalize_column_name(name: Any) -> str:
    """
    Normalize a column name.

    Column names are used only as a soft tie-breaker when selecting the best
    alignment between gold and predicted attributes. Exact column-name equality
    is not required.
    """
    return str(name).strip().lower()


def _normalize_cell_value(value: Any) -> Any:
    """
    Normalize a cell value for robust tuple comparison.

    The goal is to avoid mismatches caused only by common serialization
    differences, while preserving the represented information.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return str(value).strip().lower()

    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    if isinstance(value, str):
        return value.strip().lower()

    return str(value).strip().lower()


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicate column names while preserving the first occurrence."""
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


def _canonicalize_result(result: SQLResult) -> tuple[list[str], list[tuple[Any, ...]]]:
    """
    Convert a SQL result table into a canonical representation.

    Properties:
    - column names are normalized;
    - duplicate columns are removed, keeping the first occurrence;
    - rows are represented as tuples aligned with the canonical columns;
    - row values are normalized for robust comparison.

    Rows may be sequences or dictionaries keyed by the original column names.
    """
    if result is None:
        return [], []

    columns = result.get("columns", []) or []
    rows = result.get("rows", []) or []

    normalized_columns = [_normalize_column_name(column) for column in columns]
    unique_columns = _deduplicate_preserve_order(normalized_columns)

    first_index: dict[str, int] = {}
    for index, column in enumerate(normalized_columns):
        if column not in first_index:
            first_index[column] = index

    # Deterministic order, but not semantically binding: later we search for
    # the best alignment between gold and predicted columns.
    canonical_columns = sorted(unique_columns)

    if not rows:
        return canonical_columns, []

    canonical_rows: list[tuple[Any, ...]] = []

    if isinstance(rows[0], dict):
        for row in rows:
            values = []
            for column in canonical_columns:
                original_index = first_index[column]
                original_column_name = columns[original_index]
                values.append(_normalize_cell_value(row.get(original_column_name)))
            canonical_rows.append(tuple(values))
        return canonical_columns, canonical_rows

    for row in rows:
        values = []
        for column in canonical_columns:
            original_index = first_index[column]
            values.append(_normalize_cell_value(row[original_index]))
        canonical_rows.append(tuple(values))

    return canonical_columns, canonical_rows


def _project_relation_by_indices(
    rows: list[tuple[Any, ...]],
    indices: tuple[int, ...],
) -> Relation:
    """
    Project rows onto selected indices and return a set of tuples.

    This implements the relational set-based view: row order and duplicate rows
    are ignored.
    """
    if not indices:
        return set()

    return {tuple(row[index] for index in indices) for row in rows}


def _score_relations(
    r_pred: Relation,
    r_gold: Relation,
    *,
    alpha: float,
    beta: float,
) -> float:
    """Score two already-aligned relations."""
    if not r_gold and not r_pred:
        return 1.0

    if not r_gold and r_pred:
        return math.exp(-beta)

    if r_gold and not r_pred:
        return 0.0

    # Strict failure for single-tuple mismatches.
    # This covers scalar aggregate answers such as COUNT, AVG, MAX, MIN:
    # if both outputs contain exactly one tuple and the tuple differs,
    # there is no meaningful relational partial credit.
    if len(r_gold) == 1 and len(r_pred) == 1 and r_gold != r_pred:
        return 0.0

    missing = len(r_gold - r_pred) / len(r_gold)
    extra = len(r_pred - r_gold) / len(r_pred)

    return math.exp(-alpha * missing - beta * extra)


def _column_name_bonus(
    gold_columns: list[str],
    pred_columns: list[str],
    pred_indices: tuple[int, ...],
) -> int:
    """
    Deterministic tie-breaker for equally scoring alignments.

    It prefers mappings where normalized column names match, without making
    exact name matching mandatory.
    """
    return sum(
        1
        for gold_index, pred_index in enumerate(pred_indices)
        if gold_columns[gold_index] == pred_columns[pred_index]
    )


def relational_accuracy(
    predicted_result: SQLResult,
    gold_result: SQLResult,
    *,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> float:
    """
    Compute relation-level accuracy between predicted and gold SQL outputs.

    The metric is:

        exp(
            - alpha * |R_gold \\ R_pred| / |R_gold|
            - beta  * |R_pred \\ R_gold| / |R_pred|
        )

    where R_pred and R_gold are interpreted as sets of tuples.

    Design choices:
    - row order is ignored;
    - duplicate rows are ignored;
    - column order is ignored;
    - duplicate columns are ignored, keeping the first occurrence;
    - common serialization differences in values are normalized;
    - column names are used only as a tie-breaker;
    - the metric searches for the best injective alignment from gold columns
      to predicted columns, so aliases such as count(*) vs number_of_concerts
      do not create false negatives;
    - the predicted output is projected onto the best-matching gold arity
      before computing tuple differences;
    - extra predicted columns are penalized through the beta term, while they
      are ignored by the informational variant where beta = 0.

    This avoids two common failure modes:
    - false positives from comparing only common columns;
    - false negatives from semantically equivalent columns with different aliases.

    Edge cases:
    - both relations empty -> 1.0
    - gold empty, prediction non-empty -> exp(-beta)
    - gold non-empty, prediction empty -> 0.0
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if alpha < beta:
        raise ValueError("expected alpha >= beta")

    pred_columns, pred_rows = _canonicalize_result(predicted_result)
    gold_columns, gold_rows = _canonicalize_result(gold_result)

    n_pred = len(pred_columns)
    n_gold = len(gold_columns)

    if n_gold == 0 and n_pred == 0:
        r_gold = {tuple(row) for row in gold_rows}
        r_pred = {tuple(row) for row in pred_rows}
        return _score_relations(r_pred, r_gold, alpha=alpha, beta=beta)

    if n_gold == 0:
        r_gold = set()
        r_pred = {tuple(row) for row in pred_rows}
        return _score_relations(r_pred, r_gold, alpha=alpha, beta=beta)

    if n_pred == 0:
        r_gold = {tuple(row) for row in gold_rows}
        r_pred = set()
        return _score_relations(r_pred, r_gold, alpha=alpha, beta=beta)


    best_score = -1.0
    best_bonus = -1

    if n_pred >= n_gold:
        gold_indices = tuple(range(n_gold))
        r_gold = _project_relation_by_indices(gold_rows, gold_indices)

        extra_column_fraction = (n_pred - n_gold) / n_pred

        for pred_indices in itertools.permutations(range(n_pred), n_gold):
            r_pred = _project_relation_by_indices(pred_rows, pred_indices)

            if r_gold and not r_pred:
                score = 0.0
            elif len(r_gold) == 1 and len(r_pred) == 1 and r_gold != r_pred:
                score = 0.0
            else:
                if not r_gold and not r_pred:
                    missing = 0.0
                    row_extra = 0.0
                elif not r_gold and r_pred:
                    missing = 0.0
                    row_extra = 1.0
                else:
                    missing = len(r_gold - r_pred) / len(r_gold)
                    row_extra = len(r_pred - r_gold) / len(r_pred)

                extra = max(row_extra, extra_column_fraction)
                score = math.exp(-alpha * missing - beta * extra)

            bonus = _column_name_bonus(gold_columns, pred_columns, pred_indices)

            if score > best_score or (score == best_score and bonus > best_bonus):
                best_score = score
                best_bonus = bonus

    else:
        pred_indices = tuple(range(n_pred))
        r_pred = _project_relation_by_indices(pred_rows, pred_indices)

        missing_column_fraction = (n_gold - n_pred) / n_gold

        for gold_indices in itertools.permutations(range(n_gold), n_pred):
            r_gold = _project_relation_by_indices(gold_rows, gold_indices)

            if r_gold and not r_pred:
                score = 0.0
            elif len(r_gold) == 1 and len(r_pred) == 1 and r_gold != r_pred:
                score = 0.0
            else:
                if not r_gold and not r_pred:
                    row_missing = 0.0
                    extra = 0.0
                elif not r_gold and r_pred:
                    row_missing = 0.0
                    extra = 1.0
                else:
                    row_missing = len(r_gold - r_pred) / len(r_gold)
                    extra = len(r_pred - r_gold) / len(r_pred)

                missing = max(row_missing, missing_column_fraction)
                score = math.exp(-alpha * missing - beta * extra)

            # Here the alignment is gold -> pred, so the tie-breaker must be
            # computed in the opposite direction.
            bonus = sum(
                1
                for pred_index, gold_index in enumerate(gold_indices)
                if pred_columns[pred_index] == gold_columns[gold_index]
            )

            if score > best_score or (score == best_score and bonus > best_bonus):
                best_score = score
                best_bonus = bonus

    return float(best_score)


def informational_relational_accuracy(
    predicted_result: SQLResult,
    gold_result: SQLResult,
    *,
    alpha: float = 2.0,
) -> float:
    """
    Compute informational relational accuracy.

    This is the beta = 0 variant of relational_accuracy, hence it penalizes
    only missing gold information and does not penalize additional predicted
    tuple information.
    """
    return relational_accuracy(
        predicted_result=predicted_result,
        gold_result=gold_result,
        alpha=alpha,
        beta=0.0,
    )

# ============================================================
# Resource normalization
# ============================================================

def normalize_resource(
    value: Optional[float],
    cap: Optional[float],
    clip: bool = False,
) -> Optional[float]:
    """
    Normalize a raw resource value into [0,1].

    - None stays None
    - if cap is None or <= 0, returns None
    - values above cap are clipped to 1
    """
    if value is None:
        return None

    if cap is None or cap <= 0:
        return None

    x = max(float(value), 0.0) / float(cap)
    return min(x, 1.0) if clip else x


def collect_normalized_resources(
    resources: dict[str, Optional[float]],
    caps: Optional[dict[str, float]] = None,
    weights: Optional[dict[str, float]] = None,
) -> tuple[list[float], list[float], list[str]]:
    """
    Normalize all available resources and return:
    - normalized values
    - corresponding weights
    - resource names actually used
    """
    caps = caps or {}
    weights = weights or {}

    values = []
    used_weights = []
    names = []

    for name, raw_value in resources.items():
        norm = normalize_resource(raw_value, caps.get(name))
        if norm is None:
            continue

        values.append(norm)
        used_weights.append(float(weights.get(name, 1.0)))
        names.append(name)

    return values, used_weights, names


# ============================================================
# Aggregation functions
# ============================================================

def weighted_mean(values: list[float], weights: Optional[list[float]] = None) -> float:
    if not values:
        return 0.0

    if weights is None:
        return sum(values) / len(values)

    total_w = sum(weights)
    if total_w <= 0:
        return sum(values) / len(values)

    return sum(v * w for v, w in zip(values, weights)) / total_w


def quadratic_mean(values: list[float], weights: Optional[list[float]] = None) -> float:
    """
    Weighted RMS / quadratic mean.
    """
    if not values:
        return 0.0

    if weights is None:
        return math.sqrt(sum(v * v for v in values) / len(values))

    total_w = sum(weights)
    if total_w <= 0:
        return math.sqrt(sum(v * v for v in values) / len(values))

    return math.sqrt(sum(w * v * v for v, w in zip(values, weights)) / total_w)


def max_aggregation(values: list[float], weights: Optional[list[float]] = None) -> float:
    if not values:
        return 0.0
    return max(values)


# ============================================================
# Decay functions
# ============================================================

def exponential_decay(resource_score: float, mu: float = 1.0) -> float:
    """
    exp(- (R / mu)^2)
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")
    return math.exp(-((resource_score / mu) ** 2))


def polynomial_decay(resource_score: float, mu: float = 1.0, power: float = 2.0) -> float:
    """
    Simple polynomial-style decay clipped to [0,1]:
        max(0, 1 - (R / mu)^power)
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    val = 1.0 - (resource_score / mu) ** power
    return max(0.0, min(1.0, val))


def rational_decay(resource_score: float, mu: float = 1.0, power: float = 2.0) -> float:
    """
    1 / (1 + (R / mu)^power)
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    return 1.0 / (1.0 + (resource_score / mu) ** power)


# ============================================================
# Symbiosis score
# ============================================================

def symbiosis_score(
    *,
    accuracy: float,
    resources: Optional[dict[str, Optional[float]]] = None,
    caps: Optional[dict[str, float]] = None,
    weights: Optional[dict[str, float]] = None,
    resource_agg_fn: Optional[Callable[[list[float], Optional[list[float]]], float]] = None,
    decay_fn: Optional[Callable[..., float]] = None,
    mu: float = 1.0,
    decay_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generic symbiosis score.

    Parameters
    ----------
    accuracy:
        Main correctness/alignment score in [0,1].

    resources:
        Raw resource metrics, e.g.
        {
            "tool_calls": 3,
            "latency": None,
            "cost": None,
        }

    caps:
        Upper bounds used to normalize resources into [0,1].

    weights:
        Optional weights per resource.

    resource_agg_fn:
        Aggregation function for normalized resource values.
        Default: weighted_mean

    decay_fn:
        Decay function applied to the aggregated resource score.
        Default: exponential_decay

    mu:
        Main decay scale.

    decay_kwargs:
        Extra kwargs passed to the decay function.

    Returns
    -------
    dict with:
    - symbiosis
    - accuracy
    - resource_score
    - decay
    - normalized_resources
    - used_resources
    """
    accuracy = float(max(0.0, min(1.0, accuracy)))
    resources = resources or {}
    caps = caps or {}
    weights = weights or {}
    decay_kwargs = decay_kwargs or {}

    resource_agg_fn = resource_agg_fn or weighted_mean
    decay_fn = decay_fn or exponential_decay

    values, used_weights, used_names = collect_normalized_resources(
        resources=resources,
        caps=caps,
        weights=weights,
    )

    if values:
        resource_score = resource_agg_fn(values, used_weights)
        decay = decay_fn(resource_score, mu=mu, **decay_kwargs)
    else:
        resource_score = 0.0
        decay = 1.0

    symbiosis = accuracy * decay

    normalized_resources = {
        name: value for name, value in zip(used_names, values)
    }

    return {
        "symbiosis": float(symbiosis),
        "accuracy": accuracy,
        "resource_score": float(resource_score),
        "decay": float(decay),
        "normalized_resources": normalized_resources,
        "used_resources": used_names,
    }


# ============================================================
# Trajectory helpers
# ============================================================

def symbiosis_trajectory(
    turn_scores: list[float],
    *,
    variance_penalty_lambda: float = 0.0,
) -> dict[str, float]:
    """
    Aggregate turn-level symbiosis scores over time.

    Returns:
    - mean_symbiosis
    - variance
    - penalized_symbiosis
    """
    if not turn_scores:
        return {
            "mean_symbiosis": 0.0,
            "variance": 0.0,
            "penalized_symbiosis": 0.0,
        }

    vals = [float(x) for x in turn_scores]
    mean_val = sum(vals) / len(vals)

    if len(vals) == 1:
        var_val = 0.0
    else:
        var_val = sum((x - mean_val) ** 2 for x in vals) / len(vals)

    penalized = mean_val - variance_penalty_lambda * var_val

    return {
        "mean_symbiosis": mean_val,
        "variance": var_val,
        "penalized_symbiosis": penalized,
    }


def probability_above_threshold(
    turn_scores: list[float],
    *,
    theta: float = 0.8,
) -> float:
    """
    Estimate Pr(score >= theta) from empirical turn scores.
    """
    if not turn_scores:
        return 0.0

    vals = [float(x) for x in turn_scores]
    count = sum(1 for x in vals if x >= theta)
    return count / len(vals)


def is_highly_symbiotic(
    turn_scores: list[float],
    *,
    theta: float = 0.8,
    epsilon: float = 0.05,
) -> bool:
    """
    Check whether:
        Pr(score >= theta) >= 1 - epsilon
    """
    p = probability_above_threshold(turn_scores, theta=theta)
    return p >= (1.0 - epsilon)