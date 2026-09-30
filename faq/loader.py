import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FAQEntry:
    id: str
    question: str
    sql: str
    explanation: str = ""


def load_faqs(
    db_id: str,
    faq_dir: str | Path = "enrichment/faqs",
) -> list[FAQEntry]:
    """Load optional database-specific FAQ question/SQL pairs."""
    if not db_id or Path(db_id).name != db_id:
        raise ValueError("db_id must be a plain database identifier")

    path = Path(faq_dir) / f"{db_id}.json"
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"FAQ file must contain a JSON list: {path}")

    entries = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"FAQ entry {index} in {path} must be an object")
        missing = [key for key in ("id", "question", "sql") if not item.get(key)]
        if missing:
            raise ValueError(
                f"FAQ entry {index} in {path} is missing: {', '.join(missing)}"
            )
        entries.append(
            FAQEntry(
                id=str(item["id"]).strip(),
                question=str(item["question"]).strip(),
                sql=str(item["sql"]).strip(),
                explanation=str(item.get("explanation", "")).strip(),
            )
        )

    if len({entry.id for entry in entries}) != len(entries):
        raise ValueError(f"FAQ ids must be unique in {path}")
    return entries
