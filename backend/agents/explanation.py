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
        if rec.intent == "player_bio":
            return self._explain_player_bio(rec)
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

    @staticmethod
    def _explain_player_bio(rec: Recommendation) -> str:
        if not rec.candidates:
            return rec.headline
        player = rec.candidates[0]
        name = player.get("name", "This player")
        position = player.get("primary_position") or "baseball player"
        team = player.get("current_team") or "an MLB organization"
        age = player.get("age")
        birthplace = player.get("birthplace") or "unknown birthplace"
        bats = player.get("bats") or "unknown"
        throws = player.get("throws") or "unknown"
        debut = player.get("mlb_debut")
        teams = player.get("teams_played_recent") or []
        stats = player.get("recent_stats") or []
        news = player.get("news") or []

        parts = [
            f"{name} is a {position} for {team}.",
            f"He is {age if age else 'age unknown'} and is from {birthplace}; he bats {bats.lower()} and throws {throws.lower()}.",
        ]
        if debut:
            parts.append(f"He made his MLB debut on {debut}.")
        if teams:
            parts.append(f"Teams in his MLB record include {', '.join(teams[:5])}.")
        if stats:
            stat_bits = []
            for row in stats[:2]:
                if row.get("role") == "Batter":
                    stat_bits.append(
                        f"as a hitter, wRC+ {row.get('wRC+')}, {row.get('HR')} HR, WAR {row.get('WAR')}"
                    )
                elif row.get("role") == "Pitcher":
                    stat_bits.append(
                        f"as a pitcher, FIP {row.get('FIP')}, xFIP {row.get('xFIP')}, WAR {row.get('WAR')}"
                    )
            if stat_bits:
                parts.append("Current performance context: " + "; ".join(stat_bits) + ".")
        if news:
            headlines = [item.get("headline", "") for item in news if item.get("headline")]
            if headlines:
                parts.append("Recent news notes: " + " | ".join(headlines[:3]) + ".")
        else:
            parts.append("I did not find recent ESPN news items for him in the connected data, so this answer is based on MLB bio, team history, and stat context.")
        return " ".join(parts)

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
