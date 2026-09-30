import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.telemetry import extract_token_usage_from_message
from data import format_full_schema_from_json

from .types import ConversationState


class BaselineAgent:
    """
    Monolithic Text-to-SQL baseline.

    Features:
    - full schema in the prompt + optional semantic grounding (see below)
    - few-shot examples
    - conversational memory
    - one SQL candidate per model call
    - optional parallel sampling of N independent candidates
    """

    def __init__(
        self,
        model: Any,
        schema_json: dict,
        fewshot_retriever: Any = None,
        memory_window: int = 50,
        verbose: bool = False,
        db_id: str | None = None,
        semantic_enrichment: bool = False
    ):
        self.model = model
        self.schema_json = schema_json
        self.fewshot_retriever = fewshot_retriever
        self.memory_window = memory_window
        self.verbose = verbose
        self.db_id = db_id
        self.semantic_enrichment = semantic_enrichment

    
    def _format_schema_for_prompt(self) -> str:
        return format_full_schema_from_json(self.schema_json)

    def _get_database_specific_rules(self) -> str:
        """
        Load database-specific semantic grounding when enrichment is enabled.
        """

        if not self.semantic_enrichment:
            return "No additional database-specific semantic information."

        if not self.db_id or Path(self.db_id).name != self.db_id:
            return "No additional database-specific semantic information."

        grounding_path = (
            Path(__file__).resolve().parents[1]
            / "enrichment"
            / "grounding"
            / f"{self.db_id}.txt"
        )
        if grounding_path.exists():
            content = grounding_path.read_text(encoding="utf-8").strip()
            if content:
                return content

        return "No additional database-specific semantic information."

    def _format_memory_for_prompt(self, state: ConversationState) -> str:
        if self.memory_window <= 0:
            return "No previous conversation history."
        
        history = state.history[-self.memory_window:]

        if not history:
            return "No previous conversation history."

        lines = ["Conversation history:"]
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _build_system_prompt(self, question: str, state: ConversationState) -> str:
        schema_block = self._format_schema_for_prompt()
        memory_block = self._format_memory_for_prompt(state)
        semantic_block = self._get_database_specific_rules()

        fewshot_block = "No few-shot examples available."
        if self.fewshot_retriever is not None:
            examples = self.fewshot_retriever.retrieve(question, k=2)
            if examples:
                fewshot_block = self.fewshot_retriever.format_for_prompt(examples)

        prompt = f"""
You are a monolithic conversational Text-to-SQL agent.

Your task is to answer the user's current request by generating one valid SQL query over the database.

You have access to:
1. the full database schema
2. few-shot examples
3. recent conversation history

You must:
- use the conversation history when the request depends on previous turns
- use the full schema carefully
- return exactly one complete SQL query
- output ONLY valid JSON

Required JSON format:
{{
  "sql": "SELECT ..."
}}

Rules:
- do not use tools
- do not add explanations outside the JSON
- do not invent tables or columns
- return one complete SQL query only


{schema_block}


Database-specific semantic information:
{semantic_block}

Few-shot examples:
{fewshot_block}

{memory_block}

Current user request:
{question}
"""
        return prompt.strip()

    @staticmethod
    def _safe_json_loads(text: str) -> dict | None:
        if not isinstance(text, str):
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except Exception as e:
            print(f"Error parsing JSON: {e}")

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception as e:
                print(f"Error parsing JSON from candidate substring: {e}")

        return None

    def _parse_single_sql(self, text: str) -> str | None:
        payload = self._safe_json_loads(text)
        if not isinstance(payload, dict):
            return None

        sql = payload.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip()

        return None

    def _invoke_once(self, question: str, state: ConversationState) -> dict[str, Any]:
        system_prompt = self._build_system_prompt(question, state)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        response = self.model.invoke(messages)

        tokens_input, tokens_output = extract_token_usage_from_message(response)

        if self.verbose:
            try:
                response.pretty_print()
            except Exception:
                print(response)

        raw_text = response.content if isinstance(response.content, str) else ""
        sql = self._parse_single_sql(raw_text)

        return {
            "sql": sql,
            "raw_text": raw_text,
            "response": response,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "error": None if sql else "No valid SQL candidate parsed",
        }

    def run_turn(
        self,
        question: str,
        state: ConversationState,
    ) -> dict[str, Any]:
        """
        Generate one SQL candidate (top-1 mode).
        """
        sample = self._invoke_once(question, state)

        tokens_input = int(sample.get("tokens_input", 0) or 0)
        tokens_output = int(sample.get("tokens_output", 0) or 0)

        selected_sql = sample["sql"]

        state.history.append({"role": "user", "content": question})
        state.history.append(
            {
                "role": "assistant",
                "content": selected_sql or sample["raw_text"] or "",
            }
        )
        state.turn_index += 1

        return {
            "question": question,
            "final_answer": selected_sql or "",
            "selected_sql": selected_sql,
            "candidate_sqls": [selected_sql] if selected_sql else [],
            "selection_policy": "single_call",
            "raw_messages": [sample["response"]],
            "raw_texts": [sample["raw_text"]],
            "error": sample["error"],
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }

    def sample_candidates_parallel(
        self,
        question: str,
        state: ConversationState,
        n_samples: int = 5,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate N independent SQL candidates in parallel.

        Notes:
        - does NOT mutate conversation state
        - intended for offline baseline evaluation
        """
        start_time = perf_counter()

        if n_samples <= 0:
            return {
                "question": question,
                "selected_sql": None,
                "candidate_sqls": [],
                "selection_policy": "parallel_sampling",
                "raw_messages": [],
                "raw_texts": [],
                "error": "n_samples must be > 0",
            }

        max_workers = max_workers or n_samples

        samples = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._invoke_once, question, state)
                for _ in range(n_samples)
            ]

            for fut in as_completed(futures):
                samples.append(fut.result())

        latency_sec = perf_counter() - start_time

        candidate_sqls = []
        raw_messages = []
        raw_texts = []

        total_input_tokens = 0
        total_output_tokens = 0

        for s in samples:
            total_input_tokens += int(s.get("tokens_input", 0) or 0)
            total_output_tokens += int(s.get("tokens_output", 0) or 0)
            raw_messages.append(s["response"])
            raw_texts.append(s["raw_text"])
            if s["sql"]:
                candidate_sqls.append(s["sql"])

        selected_sql = candidate_sqls[0] if candidate_sqls else None

        return {
            "question": question,
            "selected_sql": selected_sql,
            "candidate_sqls": candidate_sqls,
            "selection_policy": "parallel_sampling",
            "raw_messages": raw_messages,
            "raw_texts": raw_texts,
            "tokens_input": total_input_tokens,
            "tokens_output": total_output_tokens,
            "latency_sec": latency_sec,
            "error": None if candidate_sqls else "No valid SQL candidates parsed",
        }
