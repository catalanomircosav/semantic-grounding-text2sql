from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConversationState:
    """
    Conversational state outside the graph.
    It is the state that the runner passes from turn to turn.
    """
    db_id: str
    turn_index: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    previous_sql: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunOutput:
    """
    An agent's output standard that can be evaluated by the runner.
    """
    user_question: str
    final_response: str

    final_sql: Optional[str] = None
    candidate_sqls: list[str] = field(default_factory=list)

    sql_result_table: Optional[dict[str, Any]] = None
    sql_explanation: Optional[str] = None


    tokens_input: int = 0
    tokens_output: int = 0
    latency_sec: float = 0.0
    estimated_cost: float = 0.0

    raw_messages: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None