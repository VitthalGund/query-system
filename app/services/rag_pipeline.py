import tempfile
import requests
import os
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


BATCH_PROMPT_TEMPLATE = """
**Role:** You are a highly specialized AI assistant for analyzing insurance policy documents. Your primary function is to provide accurate, concise, and verifiable answers based exclusively on the provided text.

**Task:** Answer every user question from the list below using ONLY the context provided.

**Instructions:**
1.  **Analyze the Context:** Carefully read all the provided document excerpts.
2.  **Strict Grounding:** Base your answers 100% on the information within the provided documents. Do NOT use any external knowledge.
3.  **Answer All Questions:** You will be given a numbered list of questions. You MUST answer every single question.
4.  **Handle Missing Information:** If the answer to a specific question cannot be found in the context, you MUST explicitly state for that question: "The provided documents do not contain information about this topic."
5.  **Critical: Use a Specific Separator:** Separate the answer for each question with the exact delimiter `---|||---`. This is essential for parsing. Do not use any other separator.

**Example Response Format:**
Answer to Question 1... ---|||--- Answer to Question 2... ---|||--- The provided documents do not contain information about this topic. ---|||--- Answer to Question 4...

**Provided Context:**
{context}

**User's Questions:**
{question}

**Expert Answers (separated by '---|||---'):**
"""
BATCH_PROMPT = PromptTemplate.from_template(BATCH_PROMPT_TEMPLATE)


async def process_url_and_questions(
    document_url: str, questions: List[str]
) -> List[str]:
    """
    (FOR API) Processes a document from a URL and answers a list of questions about it
    using a single, batched LLM call to avoid rate limiting.
    """
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            response = requests.get(document_url)
            response.raise_for_status()
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        llm = get_llm(
            provider=settings.LLM_PROVIDER, model_name=settings.LLM_MODEL_NAME
        )

        formatted_questions = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        rag_chain = (
            {
                "context": retriever | format_docs_with_metadata,
                "question": RunnablePassthrough(),
            }
            | BATCH_PROMPT
            | llm
            | StrOutputParser()
        )

        print(f"Processing {len(questions)} questions in a single batch...")

        combined_answers_str = await rag_chain.ainvoke(formatted_questions)

        answers = [ans.strip() for ans in combined_answers_str.split("---|||---")]

        if len(answers) != len(questions):
            print(
                f"Warning: Mismatch between questions ({len(questions)}) and answers ({len(answers)}). Returning raw response."
            )
            return [combined_answers_str]

        return answers

    except Exception as e:
        print(f"An error occurred in the on-the-fly RAG pipeline: {e}")
        return [f"Error processing request: {e}" for _ in questions]
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


async def process_query_from_kb(question: str) -> dict:
    """
    (FOR KNOWLEDGE BASE) Processes a single query against the pre-ingested, persistent vector store.
    Returns a detailed response with sources.
    """
    try:
        retriever = get_retriever(top_k=5)
        llm = get_llm(
            provider=settings.LLM_PROVIDER, model_name=settings.LLM_MODEL_NAME
        )

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
