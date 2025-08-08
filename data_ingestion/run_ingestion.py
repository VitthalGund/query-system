import os
import requests
from pathlib import Path
from tqdm import tqdm


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


from app.core.config import settings


def download_file(url: str, local_path: Path):
    """Downloads a file from a URL to a local path."""
    if local_path.exists():
        print(f"File {local_path} already exists. Skipping download.")
        return

    print(f"Downloading document from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        exit(1)


def create_vector_store():
    """
    Processes all documents in the source directory, creates embeddings,
    and stores them in the configured vector store (FAISS or Pinecone).
    """
    source_directory = Path("data_ingestion/source_documents")
    source_directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {source_directory}...")
    loader = DirectoryLoader(
        str(source_directory),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    documents = loader.load()
    if not documents:
        print(
            "No documents found. Please add PDF files to data_ingestion/source_documents/"
        )
        return

    print(f"Loaded {len(documents)} document pages.")

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
    )

    provider = settings.VECTOR_STORE_PROVIDER.lower()

    if provider == "faiss":
        print("Creating FAISS vector store...")
        faiss_index_path = Path(settings.FAISS_INDEX_PATH)
        faiss_index_path.mkdir(parents=True, exist_ok=True)

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(str(faiss_index_path))
        print(f"FAISS index created and saved at: {faiss_index_path}")

    elif provider == "pinecone":
        print("Creating and upserting to Pinecone vector store...")

        if not settings.PINECONE_API_KEY or not settings.PINECONE_INDEX_NAME:
            raise ValueError(
                "PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in your .env file for Pinecone integration."
            )

        print(
            f"Upserting {len(texts)} chunks to Pinecone index: '{settings.PINECONE_INDEX_NAME}'..."
        )

        PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
        )

        print("Successfully upserted documents to Pinecone.")

    else:
        raise ValueError(
            f"Unsupported vector store provider: {provider}. Please use 'faiss' or 'pinecone'."
        )


if __name__ == "__main__":
    create_vector_store()
