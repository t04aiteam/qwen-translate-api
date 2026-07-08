"""
Summarization route for Qwen3.5-4B.
Accepts both application/json and multipart/form-data.
"""

import asyncio
import os
import re
from fastapi import APIRouter, Request

from models.loader import get_llm, MODEL_ID
from schemas.request import SummarizeRequest, SummarizeResponse

router = APIRouter(tags=["Summarization"])

_LENGTH_INSTRUCTIONS = {
    "short":  "Write a short summary of 1-2 sentences only.",
    "medium": "Write a concise summary of one paragraph (4-6 sentences).",
    "long":   "Write a detailed summary of several paragraphs covering all key points.",
}

SYSTEM_PROMPT = (
    "You are a professional text summarizer. "
    "Your only task is to summarize the given text accurately and clearly. "
    "Output ONLY the summary — no preamble, no labels, no extra formatting. "
    "Preserve the original language of the text unless instructed otherwise."
)


def _build_messages(req: SummarizeRequest) -> list[dict]:
    length_instr = _LENGTH_INSTRUCTIONS.get(req.length or "medium", _LENGTH_INSTRUCTIONS["medium"])
    user_content = (
        f"Summarize the following {req.language} text.\n"
        f"{length_instr}\n\n"
        f"{req.text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*",          "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()


def _run_inference(req: SummarizeRequest) -> str:
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=_build_messages(req),
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=0.9 if req.temperature > 0 else 1.0,
    )
    return _strip_thinking(response["choices"][0]["message"]["content"])


async def _handle_summarization(req: SummarizeRequest) -> SummarizeResponse:
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, _run_inference, req)
    return SummarizeResponse(
        summary=summary,
        language=req.language or "English",
        length=req.length or "medium",
        model=os.getenv("GGUF_FILE", MODEL_ID),
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize text",
    description=(
        "Summarizes the provided `text` in its original language. "
        "Control output length with `length`: `short` (1-2 sentences), "
        "`medium` (1 paragraph), or `long` (several paragraphs). "
        "Accepts **application/json** or **multipart/form-data**."
    ),
)
async def summarize(request: Request):
    """
    Summarize text — accepts both JSON body and multipart/form-data.

    JSON example:
        {"text": "...", "language": "English", "length": "medium"}

    Form-data example:
        curl -F text="..." -F language="English" -F length="short" ...
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        req = SummarizeRequest(
            text=form.get("text"),
            language=form.get("language", "English"),
            length=form.get("length", "medium"),
            max_new_tokens=int(form.get("max_new_tokens", 512)),
            temperature=float(form.get("temperature", 0.1)),
        )
    else:
        body = await request.json()
        req = SummarizeRequest(**body)

    return await _handle_summarization(req)