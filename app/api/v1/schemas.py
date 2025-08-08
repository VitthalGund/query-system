from pydantic import BaseModel, Field
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    """
    Pydantic model for the incoming query request for the /hackrx/run endpoint.
    """

    documents: str = Field(
        ..., description="A URL pointing to a single PDF document to be processed."
    )
    questions: List[str] = Field(
        ..., description="A list of questions to be answered based on the document."
    )


class QueryResponse(BaseModel):
    """
    Pydantic model for the final API response for the /hackrx/run endpoint.
    """

    answers: List[str]


class KnowledgeBaseQueryRequest(BaseModel):
    """
    Pydantic model for querying the persistent knowledge base.
    """

    question: str = Field(..., description="The question to ask the knowledge base.")


class Source(BaseModel):
    """
    Pydantic model for a single source document chunk.
    Used for providing detailed, explainable responses.
    """

    content: str
    file_path: str | None = None
    page_number: int | None = None


class KnowledgeBaseQueryResponse(BaseModel):
    """
    A detailed response structure that includes source documents for full explainability.
    """

    question: str
    answer: str
    sources: List[Source]
