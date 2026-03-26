from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from rag.api.dependencies import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    settings = get_settings()
    if not settings.api_key:
        return "no-auth"
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
