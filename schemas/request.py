"""
Pydantic schemas for the translation endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to translate.",
        examples=["Hello, how are you?"],
    )
    input_language: str = Field(
        ...,
        description="The language of the input text (e.g. 'English', 'Vietnamese', 'French', 'Japanese').",
        examples=["English"],
    )
    target_language: str = Field(
        ...,
        description="The language to translate into (e.g. 'Vietnamese', 'Spanish', 'German').",
        examples=["Vietnamese"],
    )
    max_new_tokens: Optional[int] = Field(
        512,
        description="Maximum number of tokens to generate for the translation.",
        ge=16,
        le=4096,
    )
    temperature: Optional[float] = Field(
        0.1,
        description="Sampling temperature. Lower = more deterministic (recommended for translation).",
        ge=0.0,
        le=2.0,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The weather is beautiful today.",
                "input_language": "English",
                "target_language": "Vietnamese",
                "max_new_tokens": 256,
                "temperature": 0.1,
            }
        }
    }


class TranslateResponse(BaseModel):
    translation: str = Field(..., description="The translated text.")
    input_language: str
    target_language: str
    model: str


# ── Summarization ─────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to summarize.",
    )
    language: Optional[str] = Field(
        "English",
        description="Language of the input text. The summary will be returned in the same language.",
        examples=["English", "Vietnamese"],
    )
    length: Optional[str] = Field(
        "medium",
        description="Desired summary length: 'short' (1-2 sentences), 'medium' (1 paragraph), 'long' (several paragraphs).",
    )
    max_new_tokens: Optional[int] = Field(
        512,
        description="Maximum number of tokens to generate.",
        ge=16,
        le=4096,
    )
    temperature: Optional[float] = Field(
        0.1,
        description="Sampling temperature. Lower = more deterministic.",
        ge=0.0,
        le=2.0,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
                "language": "English",
                "length": "medium",
                "max_new_tokens": 512,
                "temperature": 0.1,
            }
        }
    }


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="The generated summary.")
    language: str
    length: str
    model: str