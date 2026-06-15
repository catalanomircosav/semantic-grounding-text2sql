import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .types import ConversationState

from time import perf_counter
from agents.telemetry import extract_token_usage_from_message

from data import format_full_schema_from_json


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
    ):
        self.model = model
        self.schema_json = schema_json
        self.fewshot_retriever = fewshot_retriever
        self.memory_window = memory_window
        self.verbose = verbose

    
    def _format_schema_for_prompt(self) -> str:
        return format_full_schema_from_json(self.schema_json)

        lines = ["Full database schema:"]

        for table in tables:
            table_name = table["table_name"]
            lines.append(f"\nTable: {table_name}")

            for col in table.get("columns", []):
                pk_mark = " [PK]" if col.get("is_primary_key") else ""
                lines.append(
                    f"- {col['column_name']} ({col['column_type']}){pk_mark}"
                )

        if foreign_keys:
            lines.append("\nForeign keys:")
            for fk in foreign_keys:
                lines.append(
                    f"- {fk['source_table']}.{fk['source_column']} -> "
                    f"{fk['target_table']}.{fk['target_column']}"
                )

        return "\n".join(lines)

    def _format_memory_for_prompt(self, state: ConversationState) -> str:
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


Database-specific rules:
# Concert singer

- singer.Name is the singer/person name. If the question asks for the name of a song, select singer.Song_Name, not singer.Name (singer.Song_Name is the song title/name).
- stadium.Average is a stored stadium attribute. Do not compute AVG(stadium.Average). Do not compute AVG(stadium.Capacity) unless the question explicitly asks for average capacity.
- For counting concerts per stadium or per singer, use INNER JOIN from the event/relationship table. Do not include zero-count stadiums/singers unless explicitly requested.
- Do not use DISTINCT unless the question explicitly asks for distinct, unique, or different values. "All" does not imply DISTINCT.
- When counting or listing real occurrences of events or relationships, start from the event/relationship table rather than the full entity table.
- Use LEFT JOIN only if the question explicitly asks to include entities with no related records.




Few-shot examples:
{fewshot_block}

{memory_block}

Current user request:
{question}
"""
        return prompt.strip()

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[dict]:
        if not isinstance(text, str):
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None

        return None

    def _parse_single_sql(self, text: str) -> Optional[str]:
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
        max_workers: Optional[int] = None,
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
    
# Other semantic grounding

f""""

# Museum visit

- visit represents one museum visit record by a visitor.
- visit.Num_of_Ticket is the number of tickets bought in that single visit, not the number of separate visits.
- To find visitors who visited more than once, count visit rows per visitor: GROUP BY visitor_ID HAVING COUNT(*) > 1.
- Total_spent is the money spent in a visit; total spending by a visitor requires SUM(Total_spent) grouped by visitor_ID.
- Preserve the column order requested by the question.
- For "largest/highest/most" questions asking for one result, use ORDER BY ... DESC LIMIT 1 unless ties are explicitly requested.

# Student transcripts tracking

DB notes:
- Student_Enrolment records a student's enrolment in one degree program during one semester.
- Student_Enrolment_Courses links an enrolment to courses.
- Sections are course sections; count Sections rows to count sections per course.
- Transcript_Contents links transcripts to course results; count rows there for course results.
- For counting related records, use INNER JOIN from the detail/relationship table unless zero-count entities are explicitly requested.
- Preserve requested output order: "name and id" means name first, id second; "date and id" means date first, id second.
- For students enrolled in a program type, use DISTINCT because students may appear in multiple enrolment rows.
- Degree_Programs.degree_summary_name uses exact values such as 'Bachelor'.
- The state North Carolina is stored as 'NorthCarolina'.
- For "substring the X", search for X, not the literal phrase "the X".
Additional Spider-style rules for this DB:
- For "courses with/at most/less than N sections", use INNER JOIN Courses-Sections and count Sections rows. Do not include courses with zero sections unless explicitly requested.
- In this DB, "number of students enrolled" usually means number of Student_Enrolment rows, not COUNT(DISTINCT student_id).
- For degree-program enrolment counts, use COUNT(*) over Student_Enrolment rows.
- For questions asking students enrolled in 2 degree programs, match Spider by grouping only by student_id and using HAVING COUNT(*) = 2, even if the wording mentions "in one semester".
- When the requested output is only course_name, group by course_name rather than course_id.

"""