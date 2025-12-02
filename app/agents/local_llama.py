"""Local Llama LLM integration for Google ADK."""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator, List

from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "llama-cpp-python is required for the local Llama service. "
        "Install it with `pip install llama-cpp-python`."
    ) from exc

SUPPORTED_MODEL_REGEX = r"local/llama-8b-gguf"


def _content_to_text(content: types.Content) -> str:
    """Extract plain text from ADK content parts."""
    text_parts: List[str] = []
    for part in content.parts:
        if part.text:
            text_parts.append(part.text)
    return "\n".join(text_parts)


def _build_messages(llm_request: LlmRequest) -> List[dict]:
    """Convert ADK request into llama.cpp chat messages."""
    messages: List[dict] = []

    system_instruction = llm_request.config.system_instruction
    if isinstance(system_instruction, str) and system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction.strip()})

    role_map = {
        "user": "user",
        "model": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    for content in llm_request.contents:
        text = _content_to_text(content)
        if not text:
            continue
        role = role_map.get(content.role or "user", "user")
        messages.append({"role": role, "content": text})

    if not messages:
        messages.append(
            {
                "role": "user",
                "content": "You are an assistant responding to an empty prompt.",
            }
        )

    return messages


class LocalLlamaLlm(BaseLlm):
    """LLM adapter that routes ADK requests to llama-cpp."""

    _llama = None
    _lock = asyncio.Lock()

    @classmethod
    def supported_models(cls) -> list[str]:
        return [SUPPORTED_MODEL_REGEX]

    @classmethod
    def _get_llama(cls) -> Llama:
        if cls._llama is not None:
            return cls._llama

        model_path = os.environ.get("LLAMA_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "LLAMA_MODEL_PATH environment variable is required for the "
                "LocalLlamaLlm service."
            )

        ctx_size = int(os.environ.get("LLAMA_CTX_SIZE", "4096"))
        gpu_layers = int(os.environ.get("LLAMA_GPU_LAYERS", "0"))

        cls._llama = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            logits_all=False,
            vocab_only=False,
        )
        return cls._llama

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        llama = self._get_llama()
        messages = _build_messages(llm_request)

        if stream:
            async with self._lock:
                iterator = await asyncio.to_thread(
                    lambda: llama.create_chat_completion(messages=messages, stream=True)
                )

            for chunk in iterator:
                delta = chunk["choices"][0]["delta"].get("content")
                if not delta:
                    continue
                content = types.Content(
                    role="model", parts=[types.Part.from_text(delta)]
                )
                yield LlmResponse(content=content, partial=True, turn_complete=False)

            yield LlmResponse(turn_complete=True)
            return

        async with self._lock:
            result = await asyncio.to_thread(
                lambda: llama.create_chat_completion(messages=messages, stream=False)
            )

        text = result["choices"][0]["message"]["content"]
        content = types.Content(role="model", parts=[types.Part.from_text(text)])
        yield LlmResponse(content=content, turn_complete=True)


LLMRegistry.register(LocalLlamaLlm)
