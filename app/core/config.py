import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    API_AUTH_TOKEN: str

    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    VECTOR_STORE_PROVIDER: str = "faiss"  # "faiss" or "pinecone"
    FAISS_INDEX_PATH: str = "vector_store_data/faiss_index"
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX_NAME: str | None = None

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
