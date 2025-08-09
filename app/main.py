import uvicorn
import os
from fastapi import FastAPI
# from .api.v1 import endpoints as v1_endpoints
from app.api.v1 import endpoints as v1_endpoints


app = FastAPI(
    title="Intelligent Query-Retrieval System API",
    description="An LLM-powered system to process documents and answer contextual questions.",
    version="1.0.0",
)


app.include_router(v1_endpoints.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {
        "message": "Welcome to the Intelligent Query-Retrieval System API. Visit /docs for details."
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))

    uvicorn.run(app, host="0.0.0.0", port=port)
