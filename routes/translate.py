"""
Translation route for Qwen3.5-4B.
Builds an instruction prompt that directs the model to act as a pure translator.

Accepts BOTH:
  - application/json      → POST body as a JSON object
  - multipart/form-data   → HTML form fields (text, input_language, target_language, ...)
"""

import os
import re
from fastapi import APIRouter, Request

from models.loader import get_llm, idle_model, MODEL_ID
from schemas.request import TranslateRequest, TranslateResponse

router = APIRouter(tags=["Translation"])

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Your only task is to translate the given text accurately and naturally. "
    "Output ONLY the translated text — no explanations, no notes, no extra formatting. "
    "Preserve the original tone, style, and formatting as closely as possible."
)


def _build_messages(req: TranslateRequest) -> list[dict]:
    user_content = (
        f"Translate the following text from {req.input_language} to {req.target_language}.\n\n"
        f"{req.text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()


def _run_inference(req: TranslateRequest) -> str:
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=_build_messages(req),
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=0.9 if req.temperature > 0 else 1.0,
    )
    return _strip_thinking(response["choices"][0]["message"]["content"])


async def _handle_translation(req: TranslateRequest) -> TranslateResponse:
    translation = await idle_model.run(_run_inference, req)
    return TranslateResponse(
        translation=translation,
        input_language=req.input_language,
        target_language=req.target_language,
        model=os.getenv("GGUF_FILE", MODEL_ID),
    )


@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate text between languages (JSON or form-data)",
    description=(
        "Translates `text` from `input_language` to `target_language` using Qwen3.5-4B. "
        "Supports 201 languages. Accepts **application/json** or **multipart/form-data**."
    ),
)
async def translate(request: Request):
    """
    Translate text — accepts both JSON body and multipart/form-data.

    JSON example:
        {"text": "Hello", "input_language": "English", "target_language": "Vietnamese"}

    Form-data example:
        curl -F text="Hello" -F input_language="English" -F target_language="Vietnamese" ...
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        req = TranslateRequest(
            text=form.get("text"),
            input_language=form.get("input_language"),
            target_language=form.get("target_language"),
            max_new_tokens=int(form.get("max_new_tokens", 512)),
            temperature=float(form.get("temperature", 0.1)),
        )
    else:
        body = await request.json()
        req = TranslateRequest(**body)

    return await _handle_translation(req)