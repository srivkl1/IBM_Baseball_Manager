"""Agent 4 - Explanation.

Takes the structured Recommendation from Agent 3 and turns it into plain-English
advice calibrated to the user's skill level. Uses the configured LLM provider
(watsonx / custom / mock).
"""
from __future__ import annotations

from typing import Optional

from backend.agents.analysis import Recommendation
from backend.baseball_knowledge import answer_basic_question
from backend.llm import LLM, get_llm


SYSTEM_TEMPLATES = {
    "beginner": (
        "You are a friendly fantasy-baseball coach. Explain recommendations using "
        "plain language. Avoid jargon. Answer naturally. For advice requests, include "
        "the recommendation and a brief why; mention risk/confidence only when useful. "
        "Use 4 short sentences max."
    ),
    "expert": (
        "You are a sharp fantasy-baseball analyst. Use advanced stats (wRC+, xFIP, "
        "Barrel%, projected PA, WAR) when they are relevant. Answer naturally instead "
        "of forcing a template; for advice requests include concrete numbers. Be concise."
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
        if rec.intent == "general_qa":
            return self._explain_general_qa(rec, skill_level)
        if rec.intent == "player_bio":
            return self._explain_player_bio(rec, skill_level)
        if rec.metrics.get("response_style") == "player_list":
            return self._explain_player_list(rec)
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

    def _explain_player_list(self, rec: Recommendation) -> str:
        if not rec.candidates:
            return rec.headline
        lines = [f"{rec.headline}:"]
        for c in rec.candidates[:10]:
            health = c.get("health", "Active")
            adjusted = c.get("health_adjusted_proj_pts", c.get("proj_pts"))
            raw = c.get("proj_pts")
            projection_text = f"{adjusted} health-adjusted pts"
            if raw is not None and adjusted != raw:
                projection_text += f" ({raw} raw)"
            health = f"{health}: {c.get('injury_note')}" if c.get("injury_note") else health
            lines.append(
                f"{c.get('rank')}. {c.get('name')} ({c.get('position')}, {c.get('team')}) - "
                f"{projection_text}, {health}, tier {c.get('tier')}."
            )
        basis = rec.metrics.get("basis")
        if basis:
            lines.append(f"Basis: {basis}.")
        return "\n".join(lines)

    def _explain_general_qa(self, rec: Recommendation, skill_level: str) -> str:
        question = rec.metrics.get("question") or rec.headline
        basic_answer = answer_basic_question(question)
        if basic_answer:
            return basic_answer
        system = (
            "You are a helpful baseball and fantasy-baseball assistant. "
            "Answer the user's question directly using your general baseball knowledge. "
            "If the question is not about baseball or fantasy baseball, answer briefly and say the app is optimized for baseball. "
            "Do not force the answer into draft, waiver, roster, or trade advice unless the user asks for that."
        )
        if skill_level == "beginner":
            system += " Use plain language and keep it concise."
        else:
            system += " You may include concise expert context when useful."
        text = self.llm.generate(
            prompt=question,
            system=system,
            max_tokens=220,
            temperature=0.2,
        )
        return text.strip()

    def _explain_player_bio(self, rec: Recommendation, skill_level: str) -> str:
        if not rec.candidates:
            return rec.headline
        player = rec.candidates[0]
        system = (
            "You are a baseball assistant. Write a natural player bio from the supplied facts. "
            "Say who the player is, what they do, age/origin/team history when available, and include fantasy/stat context briefly. "
            "Do not make up facts. If no news items are supplied, do not invent news; simply omit news or say none was found if helpful."
        )
        if skill_level == "beginner":
            system += " Keep it readable and concise."
        else:
            system += " You can include compact advanced-stat context."
        prompt = (
            f"User asked for a player bio.\n"
            f"Headline: {rec.headline}\n"
            f"Facts: {player}\n"
            f"Rationale/context bullets: {rec.rationale_bullets}\n"
            "Write the answer in 1 short paragraph or 4 compact sentences."
        )
        text = self.llm.generate(prompt=prompt, system=system, max_tokens=260, temperature=0.25)
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
