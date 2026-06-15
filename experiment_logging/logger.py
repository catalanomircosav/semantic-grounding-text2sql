from pathlib import Path
from typing import Iterable, Tuple
import json

from .schema import TurnLog


class JSONLLogger:
    def __init__(self, log_path: str | Path, auto_flush: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_flush = auto_flush
        self._file = open(self.log_path, "a", encoding="utf-8")

    def log_turn(self, turn_log: TurnLog) -> None:
        self._file.write(turn_log.to_json() + "\n")
        if self.auto_flush:
            self._file.flush()

    def log_dict(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.auto_flush:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def read_all(log_path: str | Path) -> list[dict]:
        log_path = Path(log_path)
        if not log_path.exists():
            return []

        records = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    @staticmethod
    def existing_keys(
        log_path: str | Path,
        key_fields: tuple[str, ...] = ("run_id", "db_id", "question_id", "turn_id"),
    ) -> set[Tuple]:
        """
        Returns the set of keys already present in the log file.
        Useful for skip/resume.
        """
        records = JSONLLogger.read_all(log_path)
        keys = set()

        for r in records:
            key = tuple(r.get(field) for field in key_fields)
            keys.add(key)

        return keys