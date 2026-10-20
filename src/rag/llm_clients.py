"""
src/rag/llm_clients.py
----------------------
Pluggable LLM backend:
- OllamaLLMClient: local Ollama inference.
- TemplateLLMClient: deterministic placeholder for tests and demos.

The template backend does not call a model. Its output must never
be represented as LLM-generated reasoning.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Protocol

from src.service.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_TIMEOUT_S,
    normalize_ollama_host,
)


class LLMUnavailableError(RuntimeError):
    """The model service could not be reached or timed out."""


class LLMResponseError(ValueError):
    """The service answered, but the reply was not usable."""


class LLMClient(Protocol):
    """Interface implemented by LLM backends."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the raw text response, expected to be a JSON string."""
        ...

    @property
    def backend_name(self) -> str:
        ...


class OllamaLLMClient:
    """Production backend using a locally running Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: int = DEFAULT_OLLAMA_TIMEOUT_S,
        num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
        json_mode: bool = True,
        num_predict: int = 1200,
        repeat_penalty: float = 1.1,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if num_ctx <= 0:
            raise ValueError("num_ctx must be greater than zero")
        if num_predict <= 0:
            raise ValueError("num_predict must be greater than zero")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be greater than zero")

        self.model = model
        self.host = normalize_ollama_host(host)
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.json_mode = json_mode
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty

    @property
    def backend_name(self) -> str:
        return f"ollama:{self.model}"

    def available(self, timeout: float = 2.0):
        """Return (True, '') if available, otherwise (False, reason)."""
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags",
                timeout=timeout,
            ) as resp:
                names = [
                    model.get("name", "")
                    for model in json.loads(
                        resp.read().decode("utf-8")
                    ).get("models", [])
                ]

        except (
            urllib.error.URLError,
            OSError,
            ValueError,
            TimeoutError,
        ) as exc:
            return (
                False,
                f"Ollama server not reachable at {self.host} "
                f"({type(exc).__name__})",
            )

        wanted = (
            self.model
            if ":" in self.model
            else self.model + ":"
        )

        if not any(
            name == self.model or name.startswith(wanted)
            for name in names
        ):
            return (
                False,
                f"Ollama is running but model '{self.model}' "
                f"is not pulled (run: ollama pull {self.model})",
            )

        return True, ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response through Ollama's /api/chat endpoint.

        A supplied format_schema enables schema-constrained JSON output
        when json_mode is True. Otherwise, ordinary JSON mode is used.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "repeat_penalty": self.repeat_penalty,
            },
        }

        if self.json_mode:
            payload["format"] = (
                format_schema
                if format_schema is not None
                else "json"
            )

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                body = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as exc:
            try:
                details = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )[:1000]
            except Exception:
                details = ""

            raise LLMResponseError(
                f"Ollama returned HTTP {exc.code}"
                + (f": {details}" if details else "")
            ) from exc

        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
        ) as exc:
            raise LLMUnavailableError(
                f"Ollama request failed at {self.host}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        except (ValueError, UnicodeDecodeError) as exc:
            raise LLMResponseError(
                f"Ollama reply was not valid JSON: {exc}"
            ) from exc

        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError(
                "Ollama reply has no message.content"
            ) from exc


        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                "Ollama returned an empty reply"
            )

        # Record performance statistics without logging prompt contents.
        stats = {
            key: body[key]
            for key in (
                "done_reason",
                "prompt_eval_count",
                "eval_count",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "total_duration",
            )
            if body.get(key) is not None
        }

        if stats:
            logging.getLogger(__name__).info(
                "Ollama generation stats: %s",
                stats,
            )

        return content



class GeminiLLMClient:
    """Hosted Gemini backend implementing the existing LLM interface."""

    def __init__(
        self,
        model=None,
        max_output_tokens=1200,
        timeout_seconds=180,
    ):
        import os
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        # Support Streamlit Community Cloud secrets.
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise LLMUnavailableError(
                "Gemini API key missing. Set GEMINI_API_KEY "
                "in the environment or Streamlit Secrets."
            )

        self.model = (
            model
            or os.environ.get(
                "SUBHANY_GEMINI_MODEL",
                "gemini-3.5-flash-lite",
            )
        )
        self.max_output_tokens = max_output_tokens

        try:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_seconds * 1000
                ),
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Could not initialize Gemini: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @property
    def backend_name(self) -> str:
        return f"gemini:{self.model}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        format_schema=None,
    ) -> str:
        import copy
        from google.genai import types

        config = {
            "system_instruction": system_prompt,
            "temperature": 0,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }


        if format_schema is not None:
            # Keep the original schema intact for local validation.
            gemini_schema = copy.deepcopy(format_schema)

            def sanitize_gemini_schema(node):
                if isinstance(node, dict):
                    # Gemini does not accept this JSON Schema field.
                    node.pop("additionalProperties", None)
                    node.pop("additional_properties", None)

                    # Recursively sanitize nested schemas.
                    for value in node.values():
                        sanitize_gemini_schema(value)

                elif isinstance(node, list):
                    for item in node:
                        sanitize_gemini_schema(item)

            sanitize_gemini_schema(gemini_schema)
            config["response_schema"] = gemini_schema


        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config),
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Gemini request failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        content = response.text

        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                "Gemini returned an empty response."
            )

        # Confirm that the provider returned parseable JSON.
        try:
            import json
            json.loads(content)
        except (ValueError, TypeError) as exc:
            raise LLMResponseError(
                f"Gemini returned invalid JSON: {exc}"
            ) from exc

        return content



class TemplateLLMClient:
    """
    Non-LLM stand-in for tests and offline demos.

    This produces deterministic placeholder content. It is not
    intended to provide analyst-grade security reasoning.
    """

    @property
    def backend_name(self) -> str:
        return "template"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        import re

        cluster_id_match = re.search(
            r"Cluster ID: (-?\d+)",
            user_prompt,
        )
        cluster_id = (
            cluster_id_match.group(1)
            if cluster_id_match
            else "unknown"
        )

        technique_lines = re.findall(
            r"- (T\d+(?:\.\d+)?) (.+?) \(tactic: (.+?)\) "
            r"similarity=([\d.]+) status=(\w+)",
            user_prompt,
        )

        likely_techniques = [
            {
                "technique_id": technique_id,
                "technique_name": name,
                "tactic": tactic,
                "similarity_score": float(score),
                "confidence_label": status,
                "reasoning": (
                    "[template stand-in] Listed as a "
                    f"{status.lower().replace('_', ' ')} "
                    "by the semantic mapper for this cluster's narrative."
                ),
            }
            for (
                technique_id,
                name,
                tactic,
                score,
                status,
            ) in technique_lines
        ]

        response = {
            "title": (
                f"[TEMPLATE] Cluster {cluster_id} Behavioral Summary"
            ),
            "summary": (
                "[template stand-in] This is a non-LLM-generated "
                f"placeholder report for cluster {cluster_id}, assembled "
                "directly from structured context for testing purposes only."
            ),
            "observed_behavior": (
                "[template stand-in] See structured profile and evidence "
                "in the retrieved context."
            ),
            "likely_techniques": likely_techniques,
            "risk_assessment": (
                "[template stand-in] No real risk assessment -- this report "
                "was generated by the non-LLM TemplateLLMClient for pipeline "
                "testing, not by a language model."
            ),
        }

        return json.dumps(response)
