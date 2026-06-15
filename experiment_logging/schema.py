from dataclasses import dataclass, asdict, field
from typing import Any, Optional
from datetime import datetime
import json


@dataclass
class TurnLog:
    # Experiment identity
    run_id: str
    experiment_name: str
    agent_type: str
    model_name: str

    # Datum identity
    db_id: str
    question_id: int
    turn_id: int

    # Input/output
    question: str
    model_response: str
    predicted_sql: str
    gold_sql: str

    # Execution results
    predicted_result: Optional[Any] = None
    gold_result: Optional[Any] = None

    # Metrics
    exact_match: Optional[bool] = None
    execution_match: Optional[bool] = None
    component_match: float | None = None
    oracle_match: Optional[bool] = None
    candidate_diversity: float | None = None

    # Efficiency
    tokens_input: int = 0
    tokens_output: int = 0
    latency_sec: float = 0.0
    estimated_cost: float = 0.0

    # System tracking
    num_tool_calls: int = 0
    num_tables_used: int = 0
    retrieved_tables: list[str] = field(default_factory=list)
    candidate_sqls: list[str] = field(default_factory=list)

    # Status / debug
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)