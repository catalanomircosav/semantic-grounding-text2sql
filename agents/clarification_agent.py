from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.telemetry import extract_token_usage_from_message

from .baseline_agent import BaselineAgent
from .types import ConversationState

class ClarificationAgent(BaselineAgent):
    """
    Monolithic Text-to-SQL agent with an explicit ANSWER/CLARIFY decision.
    
    The agent keeps the same context as BaselineAgent:
    - full database schema
    - optional semantic enrichment
    - few-shot examples
    - conversational memory
    
    Before producing SQL, however, it may ask one clarification question
    when the user's request is materially underspecified.
    """
    
    def _build_clarification_prompt(
        self,
        question: str,
        state: ConversationState
    ) -> str:
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

Your task is to decide whether the user's current request contains enough
information to generate a semantically well-defined SQL query.

You have access to:
1. the full database schema;
2. optonal datbase-specific semantic information;
3. few-shot Text-to-SQL examples;
4. recent conversation history.

You must choose exactly one action:

ANSWER
Use ANSWER when the request is sufficiently specified to determine the
intended database operation. Generate one complete SQL query.

CLARIFY
Use CLARIFY only when information that materially affects the intended SQL
or its result is missing or genuinely ambigous.

A clarification is appropriate when two o more plausible interpretations
of the user's request would require meaningfully different SQL queries or
would produce meaningfully different results.

Do NOT ask for clarification:
- merely because different but equivalent SQL formulations are possible;
- when the schema, semantic information, or conversation history already
  resolves the issue;
- for details that do not affects the intended query results;
- when the user's latest message resolves a clarification asked earlier.

When asking for clarification:
- ask exactly one concise question;
- ask only for the information needed to resolve the ambiguity;
- do not generate SQL.

Output ONLY valid JSON.

For ANSWER:
{{
    "action": "ANSWER",
    "clarification": null,
    "sql": "SELECT ...",
}}

For CLARIFY:
{{
    "action": "CLARIFY",
    "clarification": "concise clarification question",
    "sql": null
}}

Rules:
- action must be exactly "ANSWER" OR "CLARIFY";
- do not use tools;
- do not add explanations outside the JSON;
- do not invent tables or columns;
- use conversation history when the current message refers to previous turns;
- if the user has answered a previous clarification and the request is now
sufficiently specified, choose ANSWER.

Full database schema:
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
    
    def _parse_decision(self, text: str) -> dict[str, Any]:
        payload = self._safe_json_loads(text)
        
        if not isinstance(payload, dict):
            return {
                "action": None,
                "clarification": None,
                "sql": None,
                "error": "No valid JSON decision parsed",
            }
            
        action = payload.get("action")
        if not isinstance(action, str):
            return {
                "action": None,
                "clarification": None,
                "sql": None,
                "error": "Missing or invalid action",
            }
            
        action = action.strip().upper()
        if action == "ANSWER":
            sql = payload.get("sql")
            
            if not isinstance(sql, str) or not sql.strip():
                return {
                    "action": "ANSWER",
                    "clarification": None,
                    "sql": None,
                    "error": "ANSWER action without valid SQL",
                }
                
            return {
                "action": "ANSWER",
                "clarification": None,
                "sql": sql.strip(),
                "error": None,
            }    
        if action == "CLARIFY":
            clarification = payload.get("clarification")
            
            if not isinstance(clarification, str) or not clarification.strip():
                return {
                    "action": "CLARIFY",
                    "clarification": None,
                    "sql": None,
                    "error": "CLARIFY action without clarification question",
                }
            
            return {
                "action": "CLARIFY",
                "clarification": clarification.strip(),
                "sql": None,
                "erorr": None,
            }
        
        return {
            "action": None,
            "clarification": None,
            "sql": None,
            "error": f"Unsupported action: {action}",
        }
    
    def _invoke_decision(
        self,
        question: str,
        state: ConversationState,
    ) -> dict[str, Any]:
        system_prompt = self._build_clarification_prompt(question, state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        
        start_time = perf_counter()
        response = self.model.invoke(messages)
        latency_sec = perf_counter() - start_time
        
        tokens_input, tokens_output = extract_token_usage_from_message(response)
        
        if self.verbose:
            try:
                response.pretty_print()
            except Exception:
                print(response)
        
        raw_text = response.content if isinstance(response.content, str) else ""
        decision = self._parse_decision(raw_text)
        
        return {
            **decision,
            "raw_text": raw_text,
            "response": response,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency_sec": latency_sec
        }
        
    def run_turn(
        self,
        question: str,
        state: ConversationState,
    ) -> dict[str, Any]:
        decision = self._invoke_decision(question, state)
        
        action = decision["action"]
        sql = decision["sql"]
        clarification = decision["clarification"]
        
        if action == "ANSWER":
            assistant_content = sql or ""
        elif action == "CLARIFY":
            assistant_content = clarification or ""
        else:
            assistant_content = decision["raw_text"] or ""
        
        state.history.append(
            {
                "role": "user",
                "content": question,
            }
        )
        state.history.append(
            {
                "role":"assistant",
                "content": assistant_content,
            }
        )
        state.turn_index+=1
        
        return {
            "question": question,
            "action": action,
            "clarification": clarification,
            "sql": sql,
            "final_answer": assistant_content,
            "selected_sql": sql,
            "candidate_sqls": [sql] if sql else [],
            "selection_policy": "answer_or_clarify",
            "raw_messages": [decision["response"]],
            "raw_texts": [decision["raw_text"]],
            "tokens_input": int(decision.get("tokens_input", 0) or 0),
            "tokens_output": int(decision.get("tokens_output", 0) or 0),
            "latency_sec": float(decision.get("latency_sec", 0.0) or 0.0),
            "error": decision.get("error"),
        }
