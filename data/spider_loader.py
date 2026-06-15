import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SpiderExample:
    db_id: str
    question: str
    query: str
    question_toks: list[str] = field(default_factory=list)
    query_toks: list[str] = field(default_factory=list)


@dataclass
class SpiderSchema:
    db_id: str
    table_names_original: list[str]
    table_names: list[str]
    column_names_original: list[list[Any]]
    column_names: list[list[Any]]
    column_types: list[str]
    primary_keys: list[int]
    foreign_keys: list[list[int]]


@dataclass
class DatabaseContext:
    db_id: str
    db_path: str
    schema: SpiderSchema
    train_examples: list[SpiderExample] = field(default_factory=list)
    dev_examples: list[SpiderExample] = field(default_factory=list)

    def get_eval_turns(self, limit: Optional[int] = None) -> list[SpiderExample]:
        if limit is None:
            return self.dev_examples
        return self.dev_examples[:limit]


class SpiderDataset:
    """
    Main loader for Spider.

    It expects a structure like this:
    data/spider/
        ├── database/
        │   └── <db_id>/
        │       └── <db_id>.sqlite
        ├── train_spider.json
        ├── dev.json
        └── tables.json
    """

    def __init__(self, spider_root: str | Path):
        self.spider_root = Path(spider_root)
        self.database_dir = self.spider_root / "database"
        self.train_path = self.spider_root / "train_spider.json"
        self.dev_path = self.spider_root / "dev.json"
        self.tables_path = self.spider_root / "tables.json"

        self._validate_paths()

        self._train_raw = self._load_json(self.train_path)
        self._dev_raw = self._load_json(self.dev_path)
        self._tables_raw = self._load_json(self.tables_path)

        self.schemas_by_db = self._build_schema_index(self._tables_raw)
        self.train_examples_by_db = self._build_examples_index(self._train_raw)
        self.dev_examples_by_db = self._build_examples_index(self._dev_raw)

    def _validate_paths(self) -> None:
        missing = []
        for path in [self.database_dir, self.train_path, self.dev_path, self.tables_path]:
            if not path.exists():
                missing.append(str(path))

        if missing:
            missing_str = "\n".join(missing)
            raise FileNotFoundError(
                f"Spider dataset incomplete. Missing paths:\n{missing_str}"
            )

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_schema_index(tables_raw: list[dict]) -> dict[str, SpiderSchema]:
        schemas = {}

        for item in tables_raw:
            db_id = item["db_id"]
            schemas[db_id] = SpiderSchema(
                db_id=db_id,
                table_names_original=item.get("table_names_original", []),
                table_names=item.get("table_names", []),
                column_names_original=item.get("column_names_original", []),
                column_names=item.get("column_names", []),
                column_types=item.get("column_types", []),
                primary_keys=item.get("primary_keys", []),
                foreign_keys=item.get("foreign_keys", []),
            )

        return schemas

    @staticmethod
    def _build_examples_index(examples_raw: list[dict]) -> dict[str, list[SpiderExample]]:
        examples_by_db: dict[str, list[SpiderExample]] = {}

        for item in examples_raw:
            db_id = item["db_id"]
            example = SpiderExample(
                db_id=db_id,
                question=item["question"],
                query=item["query"],
                question_toks=item.get("question_toks", []),
                query_toks=item.get("query_toks", []),
            )

            if db_id not in examples_by_db:
                examples_by_db[db_id] = []

            examples_by_db[db_id].append(example)

        return examples_by_db

    def list_db_ids(self) -> list[str]:
        return sorted(self.schemas_by_db.keys())

    def get_schema(self, db_id: str) -> SpiderSchema:
        if db_id not in self.schemas_by_db:
            raise KeyError(f"Schema not found for db_id='{db_id}'")
        return self.schemas_by_db[db_id]

    def get_train_examples(self, db_id: str) -> list[SpiderExample]:
        return self.train_examples_by_db.get(db_id, [])

    def get_dev_examples(self, db_id: str) -> list[SpiderExample]:
        return self.dev_examples_by_db.get(db_id, [])

    def resolve_db_path(self, db_id: str) -> str:
        db_path = self.database_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(
                f"SQLite file not found for db_id='{db_id}': {db_path}"
            )
        return str(db_path)

    def build_database_context(self, db_id: str) -> DatabaseContext:
        schema = self.get_schema(db_id)
        db_path = self.resolve_db_path(db_id)
        train_examples = self.get_train_examples(db_id)
        dev_examples = self.get_dev_examples(db_id)

        return DatabaseContext(
            db_id=db_id,
            db_path=db_path,
            schema=schema,
            train_examples=train_examples,
            dev_examples=dev_examples,
        )

    def build_schema_database_json(self, db_id: str) -> dict[str, Any]:
        """
        Useful format for monolithic baseline (schema in the prompt)
        """
        schema = self.get_schema(db_id)

        tables = []
        for table_idx, table_name in enumerate(schema.table_names_original):
            columns = []

            for col_idx, (col_table_idx, col_name) in enumerate(schema.column_names_original):
                if col_table_idx != table_idx:
                    continue

                col_type = schema.column_types[col_idx] if col_idx < len(schema.column_types) else "unknown"
                is_primary = col_idx in schema.primary_keys

                columns.append(
                    {
                        "column_id": col_idx,
                        "column_name": col_name,
                        "column_type": col_type,
                        "is_primary_key": is_primary,
                    }
                )

            tables.append(
                {
                    "table_id": table_idx,
                    "table_name": table_name,
                    "columns": columns,
                }
            )

        foreign_keys = []
        for source_col_idx, target_col_idx in schema.foreign_keys:
            source_table_idx, source_col_name = schema.column_names_original[source_col_idx]
            target_table_idx, target_col_name = schema.column_names_original[target_col_idx]

            source_table = (
                schema.table_names_original[source_table_idx]
                if source_table_idx >= 0 else None
            )
            target_table = (
                schema.table_names_original[target_table_idx]
                if target_table_idx >= 0 else None
            )

            foreign_keys.append(
                {
                    "source_column_id": source_col_idx,
                    "source_table": source_table,
                    "source_column": source_col_name,
                    "target_column_id": target_col_idx,
                    "target_table": target_table,
                    "target_column": target_col_name,
                }
            )

        return {
            "db_id": db_id,
            "tables": tables,
            "foreign_keys": foreign_keys,
        }

    def build_domande_json(
        self,
        db_id: str,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Useful format for few-shot retrieval.
        """
        if split not in {"train", "dev"}:
            raise ValueError("split must be 'train' or 'dev'")

        examples = (
            self.get_train_examples(db_id)
            if split == "train"
            else self.get_dev_examples(db_id)
        )

        if limit is not None:
            examples = examples[:limit]

        return [
            {
                "db_id": ex.db_id,
                "question": ex.question,
                "query": ex.query,
                "question_toks": ex.question_toks,
                "query_toks": ex.query_toks,
            }
            for ex in examples
        ]