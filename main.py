"""
Qwen3.5-4B NLP API
Endpoints: /translate, /summarize
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models.loader import load_model, GGUF_FILE
from routes.translate import router as translate_router
from routes.summarize import router as summarize_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Qwen3.5-4B NLP API",
    description="Text translation and summarization powered by Qwen/Qwen3.5-4B-GGUF.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translate_router)
app.include_router(summarize_router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model": os.getenv("GGUF_FILE", GGUF_FILE)}


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(include_url=False)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7276)