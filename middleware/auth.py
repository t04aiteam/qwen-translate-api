"""
API key authentication dependency.
Set API_KEY in your environment (or .env). Requests must pass it
via the X-Api-Key header.
"""

import os
from fastapi import Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY", "changeme")
_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(x_api_key: str = Security(_api_key_header)):
    """Validate the API key passed in X-Api-Key header."""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it via the X-Api-Key header.",
        )
    return x_api_key