from time import perf_counter
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.telemetry import extract_token_usage_from_message
from execution import SQLExecutor

from .baseline_agent import BaselineAgent
from .types import ConversationState


class ExecutionFeedbackAgent(BaselineAgent):

    def __init__(
        self,
        *,
        model: Any,
        schema_json: dict,
        db_path: str,
        fewshot_retriever: Any = None,
        memory_window: int = 50,
        verbose: bool = False,
        db_id: Optional[str]  = None,
        semantic_enrichment: bool = False,
        feedback_row_limit: int = 20,
    ):
        super().__init__(
            model=model,
            schema_json=schema_json,
            fewshot_retriever=fewshot_retriever,
            memory_window=memory_window,
            verbose=verbose,
            db_id=db_id,
            semantic_enrichment=semantic_enrichment,
        )

        self.db_path = db_path
        self.executor = SQLExecutor()
        self.feedback_row_limit = feedback_row_limit


    def _format_execution_feedback(self, execution_result) -> str:
        """
        Convert the SQL execution result into textual feedback for the LLM.
        Only a limited number of rows is exposed to the model.
        """

        if execution_result.success:
            visible_rows = execution_result.rows[:self.feedback_row_limit]

            return (
                "Execution succeeded.\n"
                f"Columns: {execution_result.columns}\n"
                f"Rows shown: {visible_rows}\n"
                f"Total returned rows: {execution_result.row_count}\n"
                f"Execution time: {execution_result.execution_time_sec:.6f} seconds"
            )

        return (
            "Execution failed.\n"
            f"SQLite error: {execution_result.error}"
        )

    def _additional_revision_instructions(self) -> str:
        return ""

    def _revise_sql(
        self,
        *,
        question: str,
        state: ConversationState,
        initial_sql: str,
        execution_feedback: str,
    ) -> dict[str, Any]:
        system_prompt = self._build_system_prompt(question, state)
        additional_instructions = self._additional_revision_instructions()
        if additional_instructions:
            additional_instructions = f"\n\n{additional_instructions}"

        revision_prompt = f"""
        You previously generated the following SQL query:
        {initial_sql}

        The query was executed on the database.

        Execution feedback:
        {execution_feedback}

        Review the SQL query before producing the final answer.

        Evaluate the query using:
        1. the user's request;
        2. the database schema and any database-specific semantic information
        available in the system context;
        3. the execution feedback.

        Check carefully whether:
        - the selected tables and columns match the semantic meaning of the user's request;
        - joins, filters, aggregations, ordering, grouping and limits are appropriate;
        - the execution feedback reveals SQL errors, empty results, or other suspicious outcomes.{additional_instructions}

        A successful SQL execution does not by itself imply that the query is
        semantically correct.

        If the query is already correct for the user's request, return it unchanged.
        Otherwise, generate a corrected SQL query.

        Do not assume access to the reference or gold SQL query.

        Return ONLY valid JSON in this format:
        {{
            "sql": "SELECT ..."
        }}
        """.strip()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=revision_prompt),
        ]

        response = self.model.invoke(messages)

        tokens_input, tokens_output = extract_token_usage_from_message(response)

        if self.verbose:
            try:
                response.pretty_print()
            except Exception:
                print(response)

        raw_text = (
            response.content
            if isinstance(response.content, str)
            else ""
        )

        sql = self._parse_single_sql(raw_text)

        return {
            "sql": sql,
            "raw_text": raw_text,
            "response": response,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "error": (
                None
                if sql
                else "No valid revised SQL candidate parsed."
            )
        }


    def run_turn(self, question: str, state: ConversationState) -> dict[str, Any]:
        start_time = perf_counter()

        # ? QUERY GENERATION
        initial_sample = self._invoke_once(question, state)

        initial_sql = initial_sample.get("sql")

        total_input_tokens = int(initial_sample.get("tokens_input", 0) or 0)
        total_output_tokens = int(initial_sample.get("tokens_output", 0) or 0)

        raw_messages = [initial_sample["response"]]
        raw_texts = [initial_sample["raw_text"]]
        
        # ? SE NON ESISTE LA QUERY INIZIALE, MI FERMO
        if not initial_sql:
            state.history.append(
                {"role": "user", "content": question}
            )
            state.history.append(
                {
                    "role": "assistant",
                    "content": initial_sample.get("raw_text", ""),
                }
            )
            state.turn_index += 1

            latency_sec = perf_counter() - start_time

            return {
                "question": question,
                "final_answer": "",
                "selected_sql": None,
                "candidate_sqls": [],
                "selection_policy": "execution_feedback",
                "raw_messages": raw_messages,
                "raw_texts": raw_texts,
                "tokens_input": total_input_tokens,
                "tokens_output": total_output_tokens,
                "latency_sec": latency_sec,
                "num_tool_calls": 0,
                "tool_trace": [],
                "error": initial_sample.get("error")
            }

        # ? ESEGUO LA QUERY INIZIALE
        execution_result = self.executor.execute(
            self.db_path,
            initial_sql
        )

        execution_feedback = self._format_execution_feedback(
            execution_result
        )

        tool_trace = [
            {
                "sql": initial_sql,
                "success": execution_result.success,
                "columns": execution_result.columns,
                "rows": [
                    list(row)
                    for row in execution_result.rows[:self.feedback_row_limit]
                ],
                "row_count": execution_result.row_count,
                "execution_time_sec": (
                    execution_result.execution_time_sec
                ),
                "error": execution_result.error,
            }
        ]

        # ? REVISIONA SQL
        revised_sample = self._revise_sql(
            question=question,
            state=state,
            initial_sql=initial_sql,
            execution_feedback=execution_feedback,
        )

        total_input_tokens += int(revised_sample.get("tokens_input", 0) or 0)
        total_output_tokens += int(revised_sample.get("tokens_output", 0) or 0)

        raw_messages.append(revised_sample["response"])
        raw_texts.append(revised_sample["raw_text"])

        revised_sql = revised_sample.get("sql")

        # ? FALLBACK SU QUERY INIZIALE SE LA REVISIONATO NON PUO' ESSERE
        # ? PARSATA
        final_sql = revised_sql or initial_sql

        # ? AGGIORNO MEMORIA CONVERSAZIONALE
        state.history.append(
            {"role": "user", "content": question}
        )
        state.history.append(
            {
                "role": "assistant",
                "content": final_sql,
            }
        )
        state.turn_index += 1

        latency_sec = perf_counter() - start_time

        return {
            "question": question,
            "final_answer": final_sql,
            "selected_sql": final_sql,

            "candidate_sqls": [],

            "selection_policy": "execution_feedback",
            "raw_messages": raw_messages,
            "raw_texts": raw_texts,

            "tokens_input": total_input_tokens,
            "tokens_output": total_output_tokens,

            "latency_sec": latency_sec,

            "num_tool_calls": 1,
            "tool_trace": tool_trace,

            "error": (
                revised_sample.get("error")
                if revised_sql is None else None
            ),
        }
