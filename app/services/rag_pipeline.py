import tempfile
import requests
from typing import List


from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from .llm_interface import get_llm
from .vector_store import get_retriever
from app.core.config import settings


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Prepares the retrieved documents for insertion into the prompt.
    Each document is formatted with its source metadata for clear citation.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):

        source_file = (
            doc.metadata.get("source", "Unknown Source").split("/")[-1].split("\\")[-1]
        )
        page_number = doc.metadata.get("page", "N/A")

        doc_entry = (
            f"--- Document {i+1} ---\n"
            f"Source: {source_file}, Page: {page_number}\n\n"
            f"Content:\n{doc.page_content}"
        )
        formatted_docs.append(doc_entry)

    return "\n\n".join(formatted_docs)


DETAILED_PROMPT_TEMPLATE = """
**Role:** You are a highly specialized AI assistant for analyzing insurance policy documents. Your primary function is to provide accurate, concise, and verifiable answers based exclusively on the provided text.

**Task:** Answer the user's question using ONLY the context provided below.

**Instructions:**
1.  **Analyze the Context:** Carefully read all the provided document excerpts.
2.  **Strict Grounding:** Base your answer 100% on the information within the provided documents. Do NOT use any external knowledge or make assumptions.
3.  **Direct Quotation:** When possible, directly quote relevant phrases or sentences from the context to support your answer.
4.  **Cite Sources:** After each piece of information in your answer, you MUST cite the source document and page number in parentheses, like this: (Source: policy_document.pdf, Page: 5).
5.  **Synthesize Information:** If multiple documents provide relevant information, synthesize them into a coherent answer, citing each source appropriately.
6.  **Handle Missing Information:** If the answer cannot be found in the provided context, you MUST explicitly state: "The provided documents do not contain information about this topic." Do not try to guess the answer.
7.  **Be Concise:** Provide a clear and direct answer to the question. Avoid conversational filler.

**Provided Context:**
{context}

**User's Question:**
{question}

**Expert Answer:**
"""
PROMPT = PromptTemplate.from_template(DETAILED_PROMPT_TEMPLATE)


async def process_url_and_questions(
    document_url: str, questions: List[str]
) -> List[str]:
    """
    (FOR API) Processes a single document from a URL and answers multiple questions about it.
    This creates an in-memory vector store for each request.
    """
    answers = []
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            response = requests.get(document_url)
            response.raise_for_status()
            temp_file.write(response.content)
            temp_file.flush()

            loader = PyPDFLoader(temp_file.name)
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = get_llm(provider="openai", model_name="gpt-4o")

        rag_chain = (
            {
                "context": retriever | format_docs_with_metadata,
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        for question in questions:
            print(f"  - Answering (on-the-fly): '{question}'")
            answer = await rag_chain.ainvoke(question)
            answers.append(answer.strip())

        return answers

    except Exception as e:
        print(f"An error occurred in the on-the-fly RAG pipeline: {e}")
        return [f"Error processing request: {e}" for _ in questions]


async def process_query_from_kb(question: str) -> dict:
    """
    (FOR KNOWLEDGE BASE) Processes a single query against the pre-ingested, persistent vector store.
    Returns a detailed response with sources.
    """
    try:
        retriever = get_retriever(top_k=5)
        llm = get_llm(provider="openai", model_name="gpt-4o")

        rag_chain = (
            {
                "context": retriever | format_docs_with_metadata,
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        print(f"Invoking KB RAG chain for question: '{question}'")
        answer = await rag_chain.ainvoke(question)

        source_docs = retriever.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "file_path": doc.metadata.get("source", "Unknown"),
                    "page_number": doc.metadata.get("page", "N/A"),
                }
                for doc in source_docs
            ],
        }

    except Exception as e:
        print(f"Error in KB RAG pipeline: {e}")
        return {
            "error": "An error occurred while processing the query.",
            "details": str(e),
        }
