"""
utils/prompt_templates.py — Prompt engineering for LawAI India.

Provides the system prompt and context-building utilities for
multilingual (English / Hindi / Telugu) legal Q&A.
"""

from __future__ import annotations

from typing import List, Dict, Any


class PromptTemplates:
    """Collection of prompt templates for the LawAI India system."""

    # ── System prompt ─────────────────────────────────────────────────────────

    @staticmethod
    def legal_system_prompt() -> str:
        """
        Return the master system prompt for LawAI India.

        The model is instructed to:
        - answer only from retrieved context
        - cite Act name, section, and page
        - respond in the same language as the user question
        - never hallucinate legal information
        """
        return """You are LawAI India, an AI-powered legal research assistant specialising in Indian law.

CORE RULES:
1. Answer ONLY using the legal context provided below the question.
2. If the answer is not found in the retrieved documents, clearly state:
   "I could not find this information in the legal database."
3. Do NOT hallucinate, invent, or extrapolate beyond the provided context.
4. Do NOT provide personal legal advice. Provide educational information only.
5. Always cite your sources:
   - Act Name
   - Section / Article Number (if available)
   - Page Number

LANGUAGE RULE:
- If the question is in English → reply in English.
- If the question is in Hindi → reply in Hindi.
- If the question is in Telugu → reply in Telugu.
- If the question mixes languages → reply in the same mixed style.

RESPONSE FORMAT:
1. Provide a clear, accurate legal explanation based on the context.
2. Include relevant section / article references.
3. End with a "Sources" block listing every document cited.
4. Add a brief disclaimer: "This is for educational purposes only and does not constitute legal advice."

You are trusted to be precise, structured, and citation-driven."""

    # ── User prompt with context ──────────────────────────────────────────────

    @staticmethod
    def legal_user_prompt(question: str, context: str) -> str:
        """
        Build the user-turn prompt by injecting the retrieved context.

        Parameters
        ----------
        question : str
            The raw user question.
        context : str
            Pre-formatted string of retrieved legal chunks.
        """
        return f"""RETRIEVED LEGAL CONTEXT:
{context}

USER QUESTION:
{question}

Please answer the question using ONLY the legal context provided above.
Cite all relevant Acts, Sections, and Page Numbers."""

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def build_context(results: List[Dict[str, Any]]) -> str:
        """
        Convert a list of retrieval results into a formatted context string.

        Parameters
        ----------
        results : list[dict]
            Each dict has ``text``, ``score``, and ``metadata``.
        """
        if not results:
            return "No relevant legal provisions were found."

        parts: List[str] = []
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            act = meta.get("act_name", "Unknown Act")
            page = meta.get("page_number", "?")
            section = meta.get("section", "")
            chapter = meta.get("chapter", "")
            text = result.get("text", "").strip()

            header_parts = [f"[{i}] {act}", f"Page {page}"]
            if section:
                header_parts.append(f"Section {section}")
            if chapter:
                header_parts.append(f"Chapter {chapter}")

            header = " | ".join(header_parts)
            parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(parts)

    # ── Ingestion summary prompt ──────────────────────────────────────────────

    @staticmethod
    def no_context_message() -> str:
        """Standard message when no relevant chunks are found."""
        return (
            "I could not find this information in the legal database. "
            "Please ensure the relevant legal documents have been ingested. "
            "You can upload PDFs via the sidebar."
        )
