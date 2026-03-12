"""
gemini_adapters.py
──────────────────
Standalone Gemini model adapters compatible with RAGAnything / LightRAG.
Import these in any script that needs Gemini LLM, vision, or embeddings.
"""

import asyncio
import base64
import logging
import os
from typing import Optional

import google.generativeai as genai
from lightrag.utils import EmbeddingFunc

log = logging.getLogger("gemini_adapters")

# ── models ────────────────────────────────────────────────────────────────────
GEMINI_LLM_MODEL     = os.getenv("GEMINI_LLM_MODEL",    "gemini-2.0-flash")
GEMINI_VISION_MODEL  = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL   = os.getenv("GEMINI_EMBED_MODEL",  "models/text-embedding-004")
GEMINI_EMBED_DIM     = int(os.getenv("GEMINI_EMBED_DIM", "768"))


def configure(api_key: Optional[str] = None):
    """Call once before using any adapter."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY env var or pass api_key=")
    genai.configure(api_key=key)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────────────────────────
async def gemini_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    model_name: str = GEMINI_LLM_MODEL,
    **kwargs,
) -> str:
    """
    General text completion via Gemini.
    Signature matches what RAGAnything / LightRAG expects for llm_model_func.
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt or "You are a knowledgeable assistant.",
    )

    history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history_messages
    ]
    chat = model.start_chat(history=history)

    try:
        response = await asyncio.to_thread(chat.send_message, prompt)
        return response.text
    except Exception as exc:
        log.error(f"[LLM] {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Vision
# ─────────────────────────────────────────────────────────────────────────────
async def gemini_vision_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    image_data: Optional[str] = None,      # base64 JPEG/PNG
    image_mime: str = "image/jpeg",
    messages: Optional[list] = None,       # pre-formatted multimodal list
    model_name: str = GEMINI_VISION_MODEL,
    **kwargs,
) -> str:
    """
    Multimodal completion via Gemini Vision.
    Handles:
      • messages=[{"role": ..., "content": [{type,text},{type,image_url,...}]}]
      • image_data=<base64> + prompt
      • plain text prompt
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt or "You are a visual document analyst.",
    )

    # Build parts from pre-formatted messages
    if messages:
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64 = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append({"mime_type": mime, "data": base64.b64decode(b64)})
        try:
            resp = await asyncio.to_thread(model.generate_content, parts)
            return resp.text
        except Exception as exc:
            log.error(f"[Vision/messages] {exc}")
            raise

    # Build parts from image_data
    if image_data:
        parts = [{"mime_type": image_mime, "data": base64.b64decode(image_data)}, prompt]
    else:
        parts = [prompt]

    try:
        resp = await asyncio.to_thread(model.generate_content, parts)
        return resp.text
    except Exception as exc:
        log.error(f"[Vision] {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Embeddings
# ─────────────────────────────────────────────────────────────────────────────
async def gemini_embed(
    texts: list[str],
    model_name: str = GEMINI_EMBED_MODEL,
    task_type: str = "retrieval_document",
) -> list[list[float]]:
    """
    Batch text embeddings via Gemini text-embedding-004 (768-d).
    Processes texts sequentially to respect rate limits.
    """
    vectors = []
    for text in texts:
        if not text or not text.strip():
            vectors.append([0.0] * GEMINI_EMBED_DIM)
            continue
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=model_name,
                content=text[:8000],        # 8K token limit
                task_type=task_type,
            )
            vectors.append(result["embedding"])
        except Exception as exc:
            log.warning(f"[Embed] Failed for text snippet, using zeros: {exc}")
            vectors.append([0.0] * GEMINI_EMBED_DIM)
    return vectors


# ─────────────────────────────────────────────────────────────────────────────
#  EmbeddingFunc wrapper (for LightRAG / RAGAnything)
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding_func() -> EmbeddingFunc:
    """Return a LightRAG-compatible EmbeddingFunc using Gemini."""
    return EmbeddingFunc(
        embedding_dim=GEMINI_EMBED_DIM,
        max_token_size=8192,
        func=gemini_embed,
    )
