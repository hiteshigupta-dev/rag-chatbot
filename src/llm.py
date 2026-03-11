"""
LLM module.
Calls Gemini API using requests with UTF-8 safe encoding.
"""

import requests
import logging
import json
import time
from functools import lru_cache
from typing import Optional
from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def _log_llm_latency(start_time: float) -> None:
    """Log LLM generation latency."""
    logger.info("LLM generation time: %.3f seconds", time.time() - start_time)


# ---------------------------------------------------
# Text Sanitizer
# ---------------------------------------------------

def sanitize_text(text: str) -> str:
    """
    Replace Unicode smart quotes with ASCII quotes to avoid encoding errors.
    """

    if not text:
        return text

    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def sanitize_for_latin1(s: str) -> str:
    """
    Ensure a string is safe for HTTP headers (latin-1). Replace or drop chars outside range.
    """
    if not s:
        return s
    s = sanitize_text(s).strip()
    # Strip one layer of surrounding quotes (e.g. from .env pasting)
    for q in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2:
            s = s[1:-1].strip()
            break
    return "".join(c if ord(c) < 256 else "?" for c in s)


# ---------------------------------------------------
# LLM Generator
# ---------------------------------------------------

class LLMGenerator:
    """
    Handles Gemini API generation.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL_NAME):

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.api_key = api_key
        self.model_name = model_name

        # Use v1beta for generateContent (v1 has limited model/feature support)
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response using Gemini API.
        """
        prompt = sanitize_text(prompt)
        system_prompt = sanitize_text(system_prompt) if system_prompt else None

        contents = [
            {
                "parts": [{"text": prompt}]
            }
        ]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": sanitize_for_latin1(self.api_key or "")
        }

        try:
            start_time = time.time()
            payload_str = sanitize_text(json.dumps(payload, ensure_ascii=False))
            response = requests.post(
                self.api_url,
                headers=headers,
                data=payload_str.encode("utf-8"),
                timeout=30
            )

            if response.status_code != 200:
                try:
                    err_body = response.json()
                    logger.error("Gemini API error: %s - %s", response.status_code, err_body)
                except Exception:
                    logger.error("Gemini API error: %s - %s", response.status_code, response.text)
                response.raise_for_status()

            response.encoding = response.encoding or "utf-8"
            data = response.json()

            if "candidates" in data and data["candidates"]:

                candidate = data["candidates"][0]

                finish_reason = candidate.get("finishReason")

                if finish_reason == "SAFETY":
                    _log_llm_latency(start_time)
                    return "Response blocked due to safety policy."

                if finish_reason == "RECITATION":
                    _log_llm_latency(start_time)
                    return "Response blocked due to recitation policy."

                content = candidate.get("content", {})
                parts = content.get("parts", [])

                if parts and "text" in parts[0]:
                    _log_llm_latency(start_time)
                    return sanitize_text(parts[0]["text"].strip())

            logger.warning("Unexpected Gemini response format")
            _log_llm_latency(start_time)
            return "I couldn't generate a response."

        except requests.exceptions.Timeout:

            logger.error("Gemini API timeout")

            return "The language model took too long to respond."

        except requests.exceptions.HTTPError as e:

            logger.error("Gemini API HTTP error: %s", e)

            return "I'm having trouble connecting to the language model."

        except requests.exceptions.RequestException as e:

            logger.error("Gemini API request error: %s", e)

            return "I'm having trouble connecting to the language model."

        except Exception as e:

            logger.error("Unexpected LLM error: %s", e)

            return "Something went wrong while generating the response."

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using RAG context. Answer only from provided context (guardrail)."""
        if system_prompt is None:
            system_prompt = (
                "You are a Tonton FAQ support assistant. "
                "Answer ONLY from the provided context. "
                "If the answer is not in the context, respond with exactly: "
                "I'm sorry, I couldn't find this information in the FAQ. "
                "Otherwise answer clearly and concisely."
            )
        context = sanitize_text(context)
        query = sanitize_text(query)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer clearly."
        )
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=DEFAULT_LLM_TEMPERATURE,
            max_tokens=DEFAULT_LLM_MAX_TOKENS,
        )


# ---------------------------------------------------
# Global singleton
# ---------------------------------------------------

_llm_generator = None


def get_llm_generator() -> LLMGenerator:

    global _llm_generator

    if _llm_generator is None:
        _llm_generator = LLMGenerator()

    return _llm_generator


# ---------------------------------------------------
# Cached LLM response (avoids repeated API calls for same query+context)
# ---------------------------------------------------

_LLM_CACHE_MAXSIZE = 100


@lru_cache(maxsize=_LLM_CACHE_MAXSIZE)
def _cached_generate_with_context_impl(query: str, context: str, system_prompt_key: str) -> str:
    """Internal: cached wrapper. system_prompt_key is '' for default."""
    sp = None if system_prompt_key == "" else system_prompt_key
    return get_llm_generator().generate_with_context(query, context, sp)


def generate_with_context_cached(
    query: str, context: str, system_prompt: Optional[str] = None
) -> str:
    """
    Generate RAG response with LRU cache for repeated (query, context) pairs.
    Use this in the pipeline to reduce LLM latency for repeated requests.
    """
    key = system_prompt if system_prompt is not None else ""
    return _cached_generate_with_context_impl(query, context, key)


def clear_llm_response_cache() -> None:
    """Clear the LLM response cache (e.g. when user clicks Clear cache)."""
    _cached_generate_with_context_impl.cache_clear()
    logger.info("LLM response cache cleared")


# ---------------------------------------------------
# Convenience function
# ---------------------------------------------------

def generate_response(prompt: str, **kwargs) -> str:

    generator = get_llm_generator()

    return generator.generate(prompt, **kwargs)