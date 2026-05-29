"""
utils/llm.py — Ollama LLM wrapper for LawAI India.

Provides streaming token generation and conversation memory
via the Ollama HTTP API (compatible with any model served locally).
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)


class LegalLLM:
    """
    Wrapper around the Ollama HTTP API for streaming legal answers.

    Parameters
    ----------
    model_name : str
        Ollama model identifier (e.g. ``llama3.2``).
    base_url : str
        Base URL of the Ollama server.
    temperature : float
        Sampling temperature (lower = more deterministic).
    max_tokens : int
        Maximum tokens in the completion.
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._chat_url = f"{self.base_url}/api/chat"

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.

        Parameters
        ----------
        system_prompt : str
            System instruction for the model.
        user_prompt : str
            The current user question (with context injected).
        conversation_history : list, optional
            Previous messages for multi-turn memory.

        Yields
        ------
        str
            Individual text tokens as they arrive.
        """
        messages = self._build_messages(
            system_prompt, user_prompt, conversation_history or []
        )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with requests.post(
                self._chat_url,
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Run `ollama serve` and ensure the model is pulled with "
                f"`ollama pull {self.model_name}`."
            ) from exc
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out after 120 seconds.")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Non-streaming generation — returns the full response string.
        """
        return "".join(
            self.stream(system_prompt, user_prompt, conversation_history)
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Construct the Ollama messages array with conversation history.

        Only roles ``user`` and ``assistant`` are kept from history.
        System prompt always leads.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Inject prior turns (for conversation memory)
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def check_connection(self) -> bool:
        """Return True if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of locally available Ollama model names."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
