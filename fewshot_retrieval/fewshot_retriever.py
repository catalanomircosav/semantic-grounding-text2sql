import re
from dataclasses import dataclass
from typing import Optional

from data import SpiderExample


def clean_text(text: str) -> str:
    """
    Light text normalization:
    - lowercase
    - punctuation removal
    - final trim
    """
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù0-9\s]", "", text)
    return text.strip()


def jaccard_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity between token sets.
    Returns a value between 0 and 1.
    """
    a_set = set(clean_text(a).split())
    b_set = set(clean_text(b).split())

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


@dataclass
class RetrievedExample:
    example: SpiderExample
    score: float


class FewShotRetriever:
    """
    Simple lexical retriever for few-shots.

    Intended use:
    - global pool of Train Spider examples
    - top-k retrieval for question-to-question similarity
    """

    def __init__(self, examples_pool: list[SpiderExample]):
        self.examples_pool = examples_pool

    def retrieve(
        self,
        user_input: str,
        k: int = 2,
        min_score: float = 0.0,
        exclude_question: Optional[str] = None,
    ) -> list[SpiderExample]:
        """
        Returns the top-k most similar examples.

        Args:
        user_input: current user question
        k: maximum number of examples to return
        min_score: minimum similarity threshold
        exclude_question: any question to exclude (exact match)

        Returns:
        List of SpiderExamples sorted by decreasing similarity.
        """
        retrieved = self.retrieve_with_scores(
            user_input=user_input,
            k=k,
            min_score=min_score,
            exclude_question=exclude_question,
        )
        return [item.example for item in retrieved]

    def retrieve_with_scores(
        self,
        user_input: str,
        k: int = 2,
        min_score: float = 0.0,
        exclude_question: Optional[str] = None,
    ) -> list[RetrievedExample]:
        """
        Variant that also returns scores.
        Useful for debugging, analysis, and logging.
        """
        if not self.examples_pool:
            return []

        user_input_clean = clean_text(user_input)
        scored_examples: list[RetrievedExample] = []

        for ex in self.examples_pool:
            if exclude_question is not None:
                if clean_text(ex.question) == clean_text(exclude_question):
                    continue

            score = jaccard_similarity(user_input_clean, ex.question)

            if score >= min_score:
                scored_examples.append(
                    RetrievedExample(example=ex, score=score)
                )

        scored_examples.sort(key=lambda item: item.score, reverse=True)
        return scored_examples[:k]

    def format_for_prompt(
        self,
        examples: list[SpiderExample],
    ) -> str:
        """
        Converts few-shot examples into a text block to be inserted into the prompt.
        """
        if not examples:
            return "No few-shot examples available."

        formatted_blocks = []

        for idx, ex in enumerate(examples, start=1):
            block = (
                f"Example {idx}\n"
                f"Question: {ex.question}\n"
                f"SQL: {ex.query}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)