import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_openai_chat_model(
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """
    Create and return a LangChain ChatOpenAI model.

    This function:
    - loads the .env file from the project root
    - checks that OPENAI_API_KEY is available
    - supports temperature and optional max_tokens

    Args:
        model_name: OpenAI model name, e.g. "gpt-4.1-mini"
        temperature: sampling temperature
        max_tokens: optional maximum number of output tokens

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: if OPENAI_API_KEY is missing
    """

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)