import re
from dataclasses import dataclass

from .loader import FAQEntry

_STOPWORDS = {
    "a", "an", "and", "are", "by", "for", "from", "how", "in", "is",
    "of", "on", "or", "s", "the", "to", "together", "was", "were", "what",
    "which", "who", "with",
}


@dataclass(frozen=True)
class FAQMatch:
    entry: FAQEntry
    similarity: float


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ments"):
        return token[:-5]
    if len(token) > 5 and token.endswith("ment"):
        return token[:-4]
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in _STOPWORDS
    }


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def retrieve_faqs(
    question: str,
    entries: list[FAQEntry],
    *,
    threshold: float = 0.45,
    top_k: int = 1,
) -> list[FAQMatch]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    matches = [
        FAQMatch(entry, jaccard_similarity(question, entry.question))
        for entry in entries
    ]
    matches = [match for match in matches if match.similarity >= threshold]
    matches.sort(key=lambda match: (-match.similarity, match.entry.id))
    return matches[:top_k]
