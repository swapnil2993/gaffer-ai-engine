"""Custom DeepEval judge model backed by OpenRouter (OpenAI-compatible API).

DeepEval defaults its evaluation/judge model to OpenAI. This wrapper lets the
DeepEval metrics run through OpenRouter instead, reusing the same
OPENROUTER_API_KEY as the runtime engine.
"""

import os
from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_model_name() -> str:
    """OpenRouter model id for the judge (no litellm 'openrouter/' prefix)."""
    name = os.environ.get("SCOUTINTEL_EVAL_MODEL", "openai/gpt-4o")
    # Tolerate the litellm-style prefix used by the runtime engine.
    return name[len("openrouter/") :] if name.startswith("openrouter/") else name


class OpenRouterJudge(DeepEvalBaseLLM):
    """A DeepEval judge model that calls OpenRouter via the OpenAI SDK."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or _resolve_model_name()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required to use OpenRouterJudge.")
        self._client = None
        self._aclient = None

    def load_model(self, async_mode: bool = False):
        # Imported lazily so importing this module never requires openai.
        from openai import AsyncOpenAI, OpenAI

        if async_mode:
            if self._aclient is None:
                self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=OPENROUTER_BASE_URL)
            return self._aclient
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=OPENROUTER_BASE_URL)
        return self._client

    # NOTE: a custom DeepEvalBaseLLM.generate must return the VALUE directly
    # (a string, or the parsed schema object) — NOT a (value, cost) tuple. The
    # tuple convention is only for DeepEval's built-in models.
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        client = self.load_model(async_mode=False)
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            completion = client.beta.chat.completions.parse(model=self.model, messages=messages, response_format=schema)
            return completion.choices[0].message.parsed
        completion = client.chat.completions.create(model=self.model, messages=messages)
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        client = self.load_model(async_mode=True)
        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            completion = await client.beta.chat.completions.parse(
                model=self.model, messages=messages, response_format=schema
            )
            return completion.choices[0].message.parsed
        completion = await client.chat.completions.create(model=self.model, messages=messages)
        return completion.choices[0].message.content

    def get_model_name(self) -> str:
        return f"OpenRouter:{self.model}"


def get_instructor_client():
    """Returns an instructor-patched OpenAI client for structured extraction."""
    import instructor
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required.")

    client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL), mode=instructor.Mode.JSON
    )
    return client
