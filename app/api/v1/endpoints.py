from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from .schemas import (
    QueryRequest,
    QueryResponse,
    KnowledgeBaseQueryRequest,
    KnowledgeBaseQueryResponse,
)

from ..auth import get_api_key
from ...services.rag_pipeline import process_url_and_questions, process_query_from_kb

router = APIRouter()


@router.post(
    "/hackrx/run",
    response_model=QueryResponse,
    summary="Process a Single Document via URL",
    description="Processes a document from a URL and answers questions based *only* on its content. This is an on-the-fly process.",
    tags=["On-the-Fly Processing"],
)
async def run_submission(
    request: QueryRequest,
    api_key: str = Depends(get_api_key),
):
    """
    This endpoint is for on-the-fly document analysis:
    1. Authenticates the request.
    2. Downloads and processes the document from the provided URL.
    3. Answers each question based SOLELY on that document.
    4. Returns a simple list of answers as required by the problem statement.
    """
    try:
        answers = await process_url_and_questions(
            document_url=request.documents, questions=request.questions
        )
        return QueryResponse(answers=answers)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/query-kb",
    response_model=KnowledgeBaseQueryResponse,
    summary="Query the Pre-Indexed Knowledge Base",
    description="Asks a question to the persistent knowledge base of all ingested documents and returns a detailed, sourced answer.",
    tags=["Knowledge Base"],
)
async def query_knowledge_base(
    request: KnowledgeBaseQueryRequest,
    api_key: str = Depends(get_api_key),
):
    """
    This endpoint is for querying the persistent vector store created by the ingestion script:
    1. Authenticates the request.
    2. Takes a single question as input.
    3. Searches across ALL pre-ingested documents for relevant context.
    4. Returns a single, detailed answer complete with source citations for explainability.
    """
    try:
        response_data = await process_query_from_kb(request.question)

        if "error" in response_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response_data.get(
                    "details", "Failed to process query from knowledge base."
                ),
            )

        return KnowledgeBaseQueryResponse(**response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
