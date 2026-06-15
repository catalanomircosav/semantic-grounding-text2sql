from __future__ import annotations

from typing import Any


def extract_token_usage_from_message(msg: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LangChain/LangGraph message.

    Supports:
    - msg.usage_metadata
    - msg.response_metadata["token_usage"]
    """
    input_tokens = 0
    output_tokens = 0

    usage_metadata = getattr(msg, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens += int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens += int(usage_metadata.get("output_tokens", 0) or 0)
        return input_tokens, output_tokens

    response_metadata = getattr(msg, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {})
        if isinstance(token_usage, dict):
            input_tokens += int(token_usage.get("prompt_tokens", 0) or 0)
            output_tokens += int(token_usage.get("completion_tokens", 0) or 0)

    return input_tokens, output_tokens


def aggregate_token_usage(messages: list[Any]) -> tuple[int, int]:
    """
    Sum token usage across a list of messages.
    """
    total_input = 0
    total_output = 0

    for msg in messages:
        in_tok, out_tok = extract_token_usage_from_message(msg)
        total_input += in_tok
        total_output += out_tok

    return total_input, total_output