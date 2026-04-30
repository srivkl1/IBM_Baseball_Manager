"""Agent 4 - Explanation.

Takes the structured Recommendation from Agent 3 and turns it into plain-English
advice calibrated to the user's skill level. Uses the configured LLM provider
(watsonx / custom / mock).
"""
from __future__ import annotations

from typing import Optional

from backend.agents.analysis import Recommendation
from backend.llm import LLM, get_llm


SYSTEM_TEMPLATES = {
    "beginner": (
        "You are a friendly fantasy-baseball coach. Explain recommendations using "
        "plain language. Avoid jargon. Use this structure when possible: "
        "Recommendation, Why, Risk, Confidence. Use 4 short sentences max."
    ),
    "expert": (
        "You are a sharp fantasy-baseball analyst. Use advanced stats (wRC+, xFIP, "
        "Barrel%, projected PA, WAR). Use this structure when possible: "
        "Recommendation, Why, Risk, Confidence. Be concise: 4-6 sentences with concrete numbers."
    ),
}


def _prompt_from_recommendation(rec: Recommendation) -> str:
    lines = [f"Intent: {rec.intent}", f"Headline: {rec.headline}", "", "Candidates:"]
    for c in rec.candidates[:5]:
        lines.append(f" - {c}")
    lines.append("")
    lines.append("Rationale bullets:")
    for b in rec.rationale_bullets:
        lines.append(f" - {b}")
    if rec.metrics:
        lines.append("")
        lines.append(f"Metrics: {rec.metrics}")
    lines.append("")
    lines.append(
        "Please explain this recommendation to the user, including the main reason, "
        "risk, and confidence when the data supports it."
    )
    return "\n".join(lines)


class Explanation:
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or get_llm()

    def explain(self, rec: Recommendation, skill_level: str = "beginner") -> str:
        if rec.intent == "roster_lookup":
            if not rec.candidates:
                return rec.headline
            names = ", ".join(c.get("player", c.get("name", "")) for c in rec.candidates[:12])
            extra = len(rec.candidates) - 12
            if extra > 0:
                names += f", and {extra} more."
            else:
                names += "."
            return f"Here is your current roster: {names}"
        sys = SYSTEM_TEMPLATES.get(skill_level, SYSTEM_TEMPLATES["beginner"])
        prompt = _prompt_from_recommendation(rec)
        text = self.llm.generate(prompt, system=sys, max_tokens=300, temperature=0.3)
        return text.strip()

    def self_evaluate(self, rec: Recommendation, explanation: str) -> dict:
        """Simple quality check: did the explanation reference the headline pick?"""
        top = None
        if rec.candidates:
            c0 = rec.candidates[0]
            top = c0.get("name") or c0.get("team")
        mentions_top = bool(top and top.split()[-1].lower() in explanation.lower())
        long_enough = len(explanation.split()) >= 15
        return {
            "score": float(mentions_top) * 0.6 + float(long_enough) * 0.4,
            "mentions_top_candidate": mentions_top,
            "has_sufficient_detail": long_enough,
        }
