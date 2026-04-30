"""Deterministic mock LLM for offline demos and tests."""
from __future__ import annotations

from typing import Optional

from backend.baseball_knowledge import answer_basic_question

from .base import LLM


class MockLLM(LLM):
    name = "mock"

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        p = prompt.lower()
        s = (system or "").lower()

        # Explanation prompts have a structured "Candidates:" section — detect
        # and produce a plain-language paragraph mentioning the top candidate.
        if "plain language" in s or "candidates:" in p or "rationale" in p:
            top_name = self._extract_top_name(prompt)
            if top_name:
                return (
                    f"My top recommendation is {top_name}. Based on the last 3 seasons "
                    f"this profile combines steady playing time with strong projected "
                    f"fantasy points and fits your roster needs. The model leans on "
                    f"recent form plus multi-year WAR to avoid boom-or-bust picks, so "
                    f"this should give you a high floor without sacrificing upside."
                )
            return ("Based on the last 3 seasons, this pick gives you a high-floor "
                    "player with balanced category contributions and a reliable role.")

        # Intent-classification prompts (orchestrator low-confidence fallback).
        if "intent classifier" in s or "classify intent" in p:
            if any(k in p for k in ("who is on my team", "who's on my team", "my roster",
                                    "show my team", "show my roster", "who do i have",
                                    "my lineup")):
                return "intent=roster_lookup"
            if any(k in p for k in ("draft", "pick", "round")):
                return "intent=draft_pick"
            if any(k in p for k in ("trade", "waiver", "add", "drop")):
                return "intent=roster_move"
            if any(k in p for k in ("trend", "hot", "cold", "slump")):
                return "intent=player_trend"
            if any(k in p for k in ("standing", "rank", "place")):
                return "intent=standings_check"
            return "intent=general_qa"

        if "general baseball knowledge" in s:
            basic_answer = answer_basic_question(prompt)
            if basic_answer:
                return basic_answer
            return "I can answer general baseball questions, but the live LLM provider is unavailable right now."

        if "recommend" in p or "analysis" in p:
            return ("Prioritize the top projected fWAR players still on the board, "
                    "breaking ties by recent barrel rate and playing-time stability.")
        return "Acknowledged."

    @staticmethod
    def _extract_top_name(prompt: str) -> str:
        # Look for the first `'name': 'X'` or `'team': 'X'` entry in the candidates.
        import re
        for key in ("name", "team", "player"):
            m = re.search(rf"['\"]{key}['\"]:\s*['\"]([^'\"]+)['\"]", prompt)
            if m:
                return m.group(1)
        return ""
