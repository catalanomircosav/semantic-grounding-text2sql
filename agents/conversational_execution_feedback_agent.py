import re
from pathlib import Path
from time import perf_counter
from typing import Any

from faq import FAQEntry, FAQMatch, load_faqs, retrieve_faqs

from .clarification_agent import ClarificationAgent
from .execution_feedback_agent import ExecutionFeedbackAgent
from .types import ConversationState


class ConversationalExecutionFeedbackAgent(
    ClarificationAgent,
    ExecutionFeedbackAgent,
):
    """Single conversational agent with one controlled execution revision."""

    _SQLITE_UNSUPPORTED_REVISION = re.compile(r"\bILIKE\b", re.IGNORECASE)
    _REVISION_POLICIES = {"always", "on_failure_or_empty", "never"}

    def __init__(
        self,
        *,
        model: Any,
        schema_json: dict,
        db_path: str,
        fewshot_retriever: Any = None,
        memory_window: int = 50,
        verbose: bool = False,
        db_id: str | None = None,
        semantic_enrichment: bool = False,
        feedback_row_limit: int = 20,
        use_faq: bool = False,
        faq_dir: str | Path = "enrichment/faqs",
        faq_entries: list[FAQEntry] | None = None,
        faq_threshold: float = 0.45,
        faq_top_k: int = 1,
        revision_policy: str = "always",
    ):
        super().__init__(
            model=model,
            schema_json=schema_json,
            db_path=db_path,
            fewshot_retriever=fewshot_retriever,
            memory_window=memory_window,
            verbose=verbose,
            db_id=db_id,
            semantic_enrichment=semantic_enrichment,
            feedback_row_limit=feedback_row_limit,
        )
        self.use_faq = use_faq
        self.faq_threshold = faq_threshold
        self.faq_top_k = faq_top_k
        if revision_policy not in self._REVISION_POLICIES:
            supported = ", ".join(sorted(self._REVISION_POLICIES))
            raise ValueError(
                f"Unsupported revision_policy {revision_policy!r}. "
                f"Expected one of: {supported}"
            )
        self.revision_policy = revision_policy
        self.faq_entries = (
            list(faq_entries)
            if faq_entries is not None
            else load_faqs(db_id, faq_dir) if use_faq and db_id else []
        )

    def _revision_trigger(self, execution: Any) -> str | None:
        if self.revision_policy == "never":
            return None
        if self.revision_policy == "always":
            return "always"
        if not execution.success:
            return "execution_failure"
        if execution.row_count == 0:
            return "empty_result"
        if execution.rows and all(
            value is None for row in execution.rows for value in row
        ):
            return "null_only_result"
        return None

    def _faq_matches(
        self,
        question: str,
        state: ConversationState,
    ) -> list[FAQMatch]:
        if not self.use_faq:
            return []
        pending = state.metadata.get("pending_question")
        query = f"{pending} {question}" if pending else question
        return retrieve_faqs(
            query,
            self.faq_entries,
            threshold=self.faq_threshold,
            top_k=self.faq_top_k,
        )

    @staticmethod
    def _faq_block(matches: list[FAQMatch]) -> str:
        if not matches:
            return "No database-specific FAQ was retrieved."
        lines = ["Retrieved database-specific FAQs:"]
        for match in matches:
            lines.extend(
                [
                    f"- id: {match.entry.id}",
                    f"  question: {match.entry.question}",
                    f"  canonical SQL: {match.entry.sql}",
                ]
            )
            if match.entry.explanation:
                lines.append(f"  explanation: {match.entry.explanation}")
        lines.extend(
            [
                "A retrieved FAQ is only a candidate, not an automatic match.",
                "For ANSWER, add \"faq_id\": \"<id>\" only when the current "
                "request has the same semantic meaning as that FAQ; otherwise add "
                "\"faq_id\": null.",
                "If you select a FAQ, return its canonical SQL exactly.",
                "Do not select a FAQ based only on shared words, and do not use a FAQ "
                "to resolve information omitted by the user.",
            ]
        )
        return "\n".join(lines)

    def _parse_decision(self, text: str) -> dict[str, Any]:
        decision = super()._parse_decision(text)
        payload = self._safe_json_loads(text)
        faq_id = payload.get("faq_id") if isinstance(payload, dict) else None
        decision["faq_id"] = (
            faq_id.strip() if isinstance(faq_id, str) and faq_id.strip() else None
        )
        return decision

    def _build_clarification_prompt(
        self,
        question: str,
        state: ConversationState,
    ) -> str:
        prompt = super()._build_clarification_prompt(question, state)
        conservative_policy = """
Additional clarification policy for this combined agent:
- Treat a request as sufficiently specified when its intended result columns,
  entities, filters, aggregation, and grouping can be inferred from context.
- Do not ask for clarification when the request explicitly names an entity,
  category, attribute, or value.
- Do not ask whether a named value should use exact or substring matching.
- Do not ask about capitalization, singular/plural wording, or minor lexical
  variations that can be resolved from the schema and semantic information.
- CLARIFY is not a request for confirmation of SQL implementation details.
""".strip()
        return (
            f"{prompt}\n\n{conservative_policy}\n\n"
            f"{self._faq_block(self._faq_matches(question, state))}"
        )

    def _additional_revision_instructions(self) -> str:
        return """
        Before keeping the query unchanged, compare every selected column,
        aggregation, and join against each applicable database-specific semantic
        rule in the system context. In particular, distinguish stored numeric
        attributes from values that must be calculated with a SQL aggregate, and
        preserve the requested number and order of output columns.

        Do not make speculative changes that are unsupported by the user's request,
        schema, database-specific information, or execution feedback.
        """.strip()

    def _build_system_prompt(
        self,
        question: str,
        state: ConversationState,
    ) -> str:
        prompt = super()._build_system_prompt(question, state)
        return f"{prompt}\n\n{self._faq_block(self._faq_matches(question, state))}"

    @staticmethod
    def _remember(
        state: ConversationState,
        question: str,
        answer: str,
        sql: str | None = None,
    ) -> None:
        state.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        if sql:
            state.previous_sql.append(sql)
        state.turn_index += 1

    def run_turn(
        self,
        question: str,
        state: ConversationState,
    ) -> dict[str, Any]:
        start = perf_counter()
        matches = self._faq_matches(question, state)
        decision = self._invoke_decision(question, state)
        action = decision.get("action")
        initial_sql = decision.get("sql")

        if action != "ANSWER" or not initial_sql:
            clarification = decision.get("clarification")
            answer = clarification or decision.get("raw_text", "")
            if action == "CLARIFY":
                state.metadata.setdefault("pending_question", question)
            self._remember(state, question, answer)
            return {
                "question": question,
                "final_answer": answer,
                "selected_sql": None,
                "candidate_sqls": [],
                "selection_policy": "clarify_then_execution_feedback",
                "raw_messages": [decision["response"]],
                "raw_texts": [decision.get("raw_text", "")],
                "tokens_input": int(decision.get("tokens_input", 0) or 0),
                "tokens_output": int(decision.get("tokens_output", 0) or 0),
                "latency_sec": perf_counter() - start,
                "error": decision.get("error"),
                "action": action,
                "clarification": clarification,
                "initial_sql": None,
                "final_sql": None,
                "execution_feedback": None,
                "initial_execution_success": None,
                "num_tool_calls": 0,
                "num_llm_calls": 1,
                "tool_trace": [],
                "faq_id": None,
                "revision_policy": self.revision_policy,
                "revision_performed": False,
                "revision_trigger": None,
            }

        selected_faq_id = decision.get("faq_id")
        faq_match = next(
            (
                match
                for match in matches
                if match.similarity == 1.0
                or match.entry.id == selected_faq_id
            ),
            None,
        )
        if faq_match:
            initial_sql = faq_match.entry.sql

        execution = self.executor.execute(self.db_path, initial_sql)
        feedback = self._format_execution_feedback(execution)
        revision = None
        revised_sql = None
        revision_trigger = None if faq_match else self._revision_trigger(execution)
        if revision_trigger:
            revision = self._revise_sql(
                question=question,
                state=state,
                initial_sql=initial_sql,
                execution_feedback=feedback,
            )
            revised_sql = revision.get("sql")
            if (
                execution.success
                and revised_sql
                and self._SQLITE_UNSUPPORTED_REVISION.search(revised_sql)
            ):
                revised_sql = None
        final_sql = initial_sql if faq_match else revised_sql or initial_sql

        raw_messages = [decision["response"]]
        raw_texts = [decision.get("raw_text", "")]
        tokens_input = int(decision.get("tokens_input", 0) or 0)
        tokens_output = int(decision.get("tokens_output", 0) or 0)
        if revision:
            raw_messages.append(revision["response"])
            raw_texts.append(revision.get("raw_text", ""))
            tokens_input += int(revision.get("tokens_input", 0) or 0)
            tokens_output += int(revision.get("tokens_output", 0) or 0)

        state.metadata.pop("pending_question", None)
        self._remember(state, question, final_sql, final_sql)
        tool_trace = [
            {
                "sql": initial_sql,
                "success": execution.success,
                "columns": execution.columns,
                "rows": [list(row) for row in execution.rows[:self.feedback_row_limit]],
                "row_count": execution.row_count,
                "execution_time_sec": execution.execution_time_sec,
                "error": execution.error,
            }
        ]
        return {
            "question": question,
            "final_answer": final_sql,
            "selected_sql": final_sql,
            "candidate_sqls": [final_sql],
            "selection_policy": "clarify_then_execution_feedback",
            "raw_messages": raw_messages,
            "raw_texts": raw_texts,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency_sec": perf_counter() - start,
            "error": (
                None
                if faq_match or (revision and revision.get("sql"))
                else revision.get("error") if revision else None
            ),
            "action": "ANSWER",
            "clarification": None,
            "initial_sql": initial_sql,
            "final_sql": final_sql,
            "execution_feedback": feedback,
            "initial_execution_success": execution.success,
            "num_tool_calls": 1,
            "num_llm_calls": 1 + int(revision is not None),
            "tool_trace": tool_trace,
            "faq_id": faq_match.entry.id if faq_match else None,
            "revision_policy": self.revision_policy,
            "revision_performed": revision is not None,
            "revision_trigger": revision_trigger,
        }
