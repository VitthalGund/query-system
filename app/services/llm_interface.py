import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from app.core.config import settings


def get_llm(provider: str = "openai", model_name: str = "gpt-4o"):
    """
    Factory function to get a LangChain LLM instance based on the provider.

    This function initializes and returns a chat model from a specified provider.
    It handles API key validation and sets default model parameters for temperature
    and max tokens to ensure consistent behavior.

    Args:
        provider (str): The LLM provider. Supported values are 'openai', 'google', 'local'.
        model_name (str): The specific model name to use (e.g., 'gpt-4o', 'gemini-pro').

    Returns:
        A LangChain compatible LLM instance (a ChatModel).

    Raises:
        ValueError: If the provider is unsupported or if the required API key is not set.
    """
    provider = provider.lower()
    print(
        f"Initializing LLM from provider: {provider.upper()} with model: {model_name}"
    )

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=1024,
        )

    elif provider == "google":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            convert_system_message_to_human=True,
        )

    elif provider == "local":
        return Ollama(model=model_name)

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Please choose from 'openai', 'google', or 'local'."
        )
