"""Prompt and schema definitions for grounded flashcard generation."""

from textwrap import dedent
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
import json


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(
        ge=1, description="1-based page number in the source PDF.")
    chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the context chunk this citation comes from.",
    )
    char_start: Optional[int] = Field(
        default=None, ge=0, description="Start character index in the chunk (inclusive)."
    )
    char_end: Optional[int] = Field(
        default=None, ge=0, description="End character index in the chunk (exclusive)."
    )


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1, description="Verbatim snippet from the context.")
    location: Location


class Flashcard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1, description="Stable identifier for this flashcard.")
    question: str = Field(min_length=1, description="Flashcard question.")
    answer: str = Field(min_length=1, description="Grounded answer.")
    citations: List[Citation] = Field(
        min_length=1,
        description="One or more verbatim citations that support the answer.",
    )


class FlashcardSet(BaseModel):
    """Top-level structure: a batch of flashcards."""

    model_config = ConfigDict(extra="forbid")

    flashcards: List[Flashcard] = Field(
        min_length=1,
        description="List of generated flashcards.",
    )


def _strip_additional_properties(schema: dict) -> dict:
    """
    Recursively remove additionalProperties/additional_properties so GenAI schema validation accepts it.
    """
    if not isinstance(schema, dict):
        return schema
    schema.pop("additionalProperties", None)
    schema.pop("additional_properties", None)
    for key, value in list(schema.items()):
        if isinstance(value, dict):
            schema[key] = _strip_additional_properties(value)
        elif isinstance(value, list):
            schema[key] = [_strip_additional_properties(v) for v in value]
    return schema


class FlashcardSetStrict(FlashcardSet):
    """Schema sanitized for response_schema compatibility."""

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        return _strip_additional_properties(schema)


class CoverageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continue_: bool = Field(
        alias="continue", description="Whether to continue generating.")
    coverage: float = Field(ge=0, le=1, description="Coverage ratio achieved.")
    missing: List[str] = Field(
        default_factory=list, description="Chunk IDs not yet cited.")


class CoverageResultStrict(CoverageResult):
    """Schema sanitized for response_schema compatibility."""

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        return _strip_additional_properties(schema)


FLASHCARD_SYSTEM_PROMPT = dedent(
    f"""
    You are an agent that generates study flashcards from provided context chunks.

    When a user asks you to create flashcards:
    1. Read the user's request and the provided context chunks (each chunk includes a `chunk_id` and `page`).
    2. Use **only** the information contained in these context chunks. Do not use outside knowledge or make unsupported inferences.
    3. If the context is insufficient to create accurate flashcards for some or all of the request, explicitly state that the context is insufficient instead of guessing.
    4. For every flashcard you create:
    - Base the flashcard content strictly on the context.
    - Include at least one verbatim supporting span from the context.
    - For each quoted span, record the associated `chunk_id` and `page`.
    5. Respond ONLY with a JSON object matching this exact schema:
    {json.dumps(FlashcardSetStrict.model_json_schema(), indent=2)}
    """
).strip()

REFINER_SYSTEM_PROMPT = dedent(
    """
    You are a flashcards coverage refiner.

    Flashcards draft: {flashcards}
    Coverage Report: {coverage_result}

    Your task is to analyze the coverage_result.
    - IF the `continue_|continue` key is EXACTLY "False", you MUST call the `exit_loop()` function and nothing else.
    - OTHERWISE, refine the draft using {total_chunks} to resolve the IDs in the `missing` key from the coverage_result.
    """
).strip()

COVERAGE_SYSTEM_PROMPT = dedent(
    """
    You are a coverage evaluator for flashcards.

    1. Read the flashcards from {{flashcards}}
    2. Read the `total_chunks` from {{total_chunks}}
    3. Use the `extract_cited_ids()` tool to extract the chunks IDs (`cited_chunks`) 
    
    With:
    - total_chunks: all chunk IDs retrieved
    - cited_chunks: chunk IDs cited by generated flashcards so far
    - coverage_threshold (0-1): {coverage_threshold}

    4. Decide whether to continue generation.
    5. Respond ONLY with a JSON object matching this exact schema:
    {coverage_schema}
    6. Only include missing chunk_ids that are not yet cited.
    """
).strip()
