from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency to validate the API key from the Authorization header.
    The expected format is "Bearer <your_token>".
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )

    parts = api_key.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Must be 'Bearer <token>'",
        )

    token = parts[1]

    if token == settings.API_AUTH_TOKEN:
        return token
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Invalid API Key.",
        )
