import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExecutionResult:
    success: bool
    sql: str
    db_path: str

    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0

    execution_time_sec: float = 0.0
    error: Optional[str] = None

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "sql": self.sql,
            "db_path": self.db_path,
            "rows": [list(row) for row in self.rows],
            "columns": self.columns,
            "row_count": self.row_count,
            "execution_time_sec": self.execution_time_sec,
            "error": self.error,
        }


class SQLExecutor:
    """
    Robust SQLite executor for Spider.

    Typical use:
        executor = SQLExecutor()
        result = executor.execute(db_path, "SELECT * FROM singer")
    """

    def __init__(self, fetch_limit: Optional[int] = None):
        """
        Args:
            fetch_limit:
                Max number of lines to read.
                If None, it reads everything.
        """
        self.fetch_limit = fetch_limit

    def execute(self, db_path: str | Path, sql: str) -> ExecutionResult:
        start_time = time.perf_counter()
        db_path = str(db_path)

        if not sql or not sql.strip():
            return ExecutionResult(
                success=False,
                sql=sql,
                db_path=db_path,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_sec=0.0,
                error="Empty SQL query",
            )

        if not Path(db_path).exists():
            return ExecutionResult(
                success=False,
                sql=sql,
                db_path=db_path,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_sec=0.0,
                error=f"Database file not found: {db_path}",
            )

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            # ! impedisco modifiche al database (per sicurezza)
            conn.execute("PRAGMA query_only = ON")

            cursor = conn.cursor()

            cursor.execute(sql)

            columns = []
            if cursor.description is not None:
                columns = [col[0] for col in cursor.description]

            if self.fetch_limit is None:
                rows = cursor.fetchall()
            else:
                rows = cursor.fetchmany(self.fetch_limit)

            execution_time_sec = time.perf_counter() - start_time

            return ExecutionResult(
                success=True,
                sql=sql,
                db_path=db_path,
                rows=rows,
                columns=columns,
                row_count=len(rows),
                execution_time_sec=execution_time_sec,
                error=None,
            )

        except Exception as e:
            execution_time_sec = time.perf_counter() - start_time

            return ExecutionResult(
                success=False,
                sql=sql,
                db_path=db_path,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_sec=execution_time_sec,
                error=str(e),
            )

        finally:
            if conn is not None:
                conn.close()