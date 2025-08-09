import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional


load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    PORT: int
    API_AUTH_TOKEN: str

    LLM_PROVIDER: str = "openai"
    LLM_MODEL_NAME: str = "gpt-4o"

    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    VECTOR_STORE_PROVIDER: str = "faiss"
    FAISS_INDEX_PATH: str = "vector_store_data/faiss_index"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
