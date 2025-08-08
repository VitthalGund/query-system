from pathlib import Path
from app.core.config import settings


from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_retriever(top_k: int = 5) -> VectorStoreRetriever:
    """
    Initializes and returns a retriever from the configured vector store.
    This function is for loading an existing index, not creating one.

    Args:
        top_k (int): The number of relevant documents to retrieve.

    Returns:
        A LangChain VectorStoreRetriever object.
    """
    provider = settings.VECTOR_STORE_PROVIDER.lower()

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    print(f"Initializing retriever from provider: {provider.upper()}")

    if provider == "faiss":
        faiss_index_path = Path(settings.FAISS_INDEX_PATH)
        if (
            not faiss_index_path.exists()
            or not (faiss_index_path / "index.faiss").exists()
        ):
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_index_path}. "
                "Please run the ingestion script (data_ingestion/run_ingestion.py) first."
            )

        vector_store = FAISS.load_local(
            str(faiss_index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("FAISS retriever loaded successfully.")

    elif provider == "pinecone":

        if not settings.PINECONE_API_KEY or not settings.PINECONE_INDEX_NAME:
            raise ValueError(
                "PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in your .env file for Pinecone integration."
            )

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME, embedding=embeddings
        )
        print("Pinecone retriever loaded successfully.")

    else:
        raise ValueError(
            f"Unsupported vector store provider: '{provider}'. Please use 'faiss' or 'pinecone'."
        )

    return vector_store.as_retriever(search_kwargs={"k": top_k})
