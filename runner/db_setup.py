from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data import (
    SpiderDataset,
    DatabaseContext,
)
from fewshot_retrieval import FewShotRetriever

@dataclass
class DatabasePackage:
    """
    Complete package for working on a single DB.
    """
    context: DatabaseContext
    fewshot_retriever: FewShotRetriever
    eval_turns: list


def _build_global_train_pool(dataset: SpiderDataset):
    all_train_examples = []
    for db_id in dataset.list_db_ids():
        all_train_examples.extend(dataset.get_train_examples(db_id))
    return all_train_examples


def prepare_database(
    db_id: str,
    spider_path: str | Path = "../data/spider",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    eval_start: int = 0,
    eval_limit: Optional[int] = 50,
) -> DatabasePackage:
    """
    Prepare everything needed to work on a database.

    Steps:
    1. Load Spider
    2. Build DatabaseContext
    3. Build global few-shot retriever
    4. Prepare evaluation rounds

    Returns:
    DatabasePackage ready for agents and runners
    """

    # ---------------------------
    # 1. Dataset
    # ---------------------------
    dataset = SpiderDataset(spider_path)

    # ---------------------------
    # 2. Database context
    # ---------------------------
    context = dataset.build_database_context(db_id)

    # ---------------------------
    # 3. Few-shot retriever (global pool)
    # ---------------------------
    train_pool = _build_global_train_pool(dataset)

    fewshot_retriever = FewShotRetriever(train_pool)

    # ---------------------------
    # 4. Evaluation turns (conversation)
    # ---------------------------
    if eval_start < 0:
        raise ValueError("eval_start must be >= 0")
    
    if  eval_limit is not None and eval_limit < 0:
        raise ValueError("eval_limit must be >= 0 or None")
    
    all_eval_turns = context.get_eval_turns(limit=None)

    if eval_limit is None:
        eval_turns = all_eval_turns[eval_start:]
    else:
        eval_turns = all_eval_turns[
            eval_start : eval_start + eval_limit
        ]
    
    # ---------------------------
    # Final package
    # ---------------------------
    return DatabasePackage(
        context=context,
        fewshot_retriever=fewshot_retriever,
        eval_turns=eval_turns,
    )