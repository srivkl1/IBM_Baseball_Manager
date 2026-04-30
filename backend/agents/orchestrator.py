"""Agent 1 - Orchestrator.

Responsibilities:
  1. Interpret the user request.
  2. Ask a clarifying question if intent is unclear.
  3. Assemble a structured data request for downstream agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.llm import LLM, get_llm

INTENTS = {
    "draft_pick": "Recommend the next pick in an active fantasy draft.",
    "roster_lookup": "Show the current roster for the user's fantasy team.",
    "roster_move": "Recommend an add/drop, trade, or waiver move.",
    "waiver_scan": "Scan the league free-agent pool for useful adds.",
    "team_diagnosis": "Summarize roster strengths, weaknesses, and strategic needs.",
    "trade_analysis": "Find balanced trade ideas or likely trade partners.",
    "lineup_optimization": "Suggest starters, bench candidates, and roster slot priorities.",
    "risk_check": "Identify risky players due to injury, role, production, or playing-time concerns.",
    "player_bio": "Explain who a player is using bio, career, fantasy, and news context.",
    "player_trend": "Explain recent performance trend for a player.",
    "standings_check": "Report current standings or season progress.",
}


@dataclass
class AgentRequest:
    user_text: str
    skill_level: str = "beginner"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorPlan:
    intent: str
    confidence: float
    clarification: Optional[str]
    data_request: Dict[str, Any]


def _rule_based_intent(text: str) -> tuple[str, float]:
    t = text.lower()
    if any(k in t for k in ("risk", "risky", "injury", "injured", "playing time", "role security")):
        return "risk_check", 0.82
    if any(k in t for k in ("lineup", "start", "bench", "optimize")):
        return "lineup_optimization", 0.82
    if any(k in t for k in (
        "who is on my team", "who's on my team", "my roster", "show my team",
        "show my roster", "who do i have", "my lineup", "who is on joseph",
        "who's on joseph", "who is on joseph's", "who's on joseph's",
    )):
        return "roster_lookup", 0.95
    if any(k in t for k in ("who is ", "who's ", "tell me about ", "bio", "background", "where is")):
        return "player_bio", 0.88
    if any(k in t for k in ("draft", "pick", "round", "on the clock", "bpa")):
        return "draft_pick", 0.85
    if any(k in t for k in ("trade", "swap", "offer", "deal")):
        return "trade_analysis", 0.88
    if any(k in t for k in ("waiver", "free agent", "free-agent", "available player", "wire")):
        return "waiver_scan", 0.86
    if any(k in t for k in ("add", "drop", "replace")):
        return "roster_move", 0.80
    if any(k in t for k in ("diagnose", "team needs", "weakness", "strength", "where can i improve")):
        return "team_diagnosis", 0.84
    if any(k in t for k in ("trend", "hot", "cold", "slump", "streak")):
        return "player_trend", 0.75
    if any(k in t for k in (
        "standing", "rank", "place", "leader", "where do i stand",
        "how am i doing", "my team doing",
    )):
        return "standings_check", 0.80
    return "draft_pick", 0.40


class Orchestrator:
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or get_llm()

    def plan(self, req: AgentRequest) -> OrchestratorPlan:
        intent, conf = _rule_based_intent(req.user_text)

        if conf < 0.6:
            sys = ("You are an intent classifier for a fantasy-baseball assistant. "
                   f"Valid intents: {list(INTENTS)}. "
                   "Reply ONLY as: intent=<one of them>.")
            reply = self.llm.generate(
                f"Classify intent: {req.user_text}",
                system=sys,
                max_tokens=32,
                temperature=0,
            )
            for key in INTENTS:
                if key in reply:
                    intent, conf = key, 0.65
                    break

        clarification = None
        if conf < 0.5:
            clarification = (
                "I want to make sure I get this right - are you asking about a draft "
                "pick, your roster, a trade, waiver move, risk check, a player's recent trend, "
                "or your current standings?"
            )

        data_request = {
            "intent": intent,
            "needs_player_pool": intent in (
                "draft_pick", "roster_move", "waiver_scan", "team_diagnosis",
                "trade_analysis", "lineup_optimization", "risk_check",
            ),
            "needs_recent_form": intent in (
                "player_bio", "player_trend", "roster_move", "waiver_scan", "team_diagnosis",
                "trade_analysis", "lineup_optimization", "risk_check",
            ),
            "needs_standings": intent == "standings_check",
            "user_context": req.context,
            "user_text": req.user_text,
        }
        return OrchestratorPlan(
            intent=intent,
            confidence=conf,
            clarification=clarification,
            data_request=data_request,
        )
