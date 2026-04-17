"""End-to-end agentic pipeline: orchestrator -> retrieval -> analysis -> explanation.

The workflow returns an `AgentResponse` containing every intermediate artifact,
so the Streamlit UI can visualize the full trace for students/judges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.agents.orchestrator import (AgentRequest, Orchestrator,
                                         OrchestratorPlan)
from backend.agents.data_retrieval import DataRetrieval
from backend.agents.analysis import Analysis, Recommendation
from backend.agents.explanation import Explanation
from backend.draft import simulator as sim
from backend.llm import LLM, get_llm


@dataclass
class AgentResponse:
    plan: OrchestratorPlan
    data_bundle_summary: Dict[str, Any]
    recommendation: Recommendation
    explanation: str
    self_eval: Dict[str, Any]
    trace: Dict[str, Any] = field(default_factory=dict)


def _summarize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    summary = {"data_source": bundle.get("data_source")}
    for key in ("player_pool", "recent_batting", "recent_pitching",
                "prospects"):
        df = bundle.get(key)
        if df is not None:
            summary[f"{key}_rows"] = int(len(df))
    lg = bundle.get("league")
    if lg is not None:
        summary["league"] = {"source": lg.source, "teams": len(lg.teams),
                             "scoring": lg.scoring_type,
                             "season": lg.season}
    return summary


class AgentPipeline:
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or get_llm()
        self.orchestrator = Orchestrator(self.llm)
        self.retrieval = DataRetrieval()
        self.analysis = Analysis()
        self.explanation = Explanation(self.llm)

    def run(self, user_text: str, skill_level: str = "beginner",
            context: Optional[Dict[str, Any]] = None,
            draft_state: Optional[sim.DraftState] = None,
            standings_table=None) -> AgentResponse:
        req = AgentRequest(user_text=user_text, skill_level=skill_level,
                           context=context or {})
        plan = self.orchestrator.plan(req)

        if plan.clarification:
            # Short-circuit: return the clarification as the explanation.
            empty_rec = Recommendation(intent=plan.intent,
                                       headline="Need more detail.")
            return AgentResponse(
                plan=plan,
                data_bundle_summary={},
                recommendation=empty_rec,
                explanation=plan.clarification,
                self_eval={"score": 1.0, "clarification": True},
                trace={"provider": self.llm.name},
            )

        bundle = self.retrieval.fetch(plan.data_request)
        if standings_table is not None:
            bundle["standings_table"] = standings_table
        rec = self.analysis.analyze(plan.intent, bundle, draft_state=draft_state)
        explanation = self.explanation.explain(rec, skill_level=skill_level)
        self_eval = self.explanation.self_evaluate(rec, explanation)

        return AgentResponse(
            plan=plan,
            data_bundle_summary=_summarize_bundle(bundle),
            recommendation=rec,
            explanation=explanation,
            self_eval=self_eval,
            trace={"provider": self.llm.name,
                   "data_source": bundle.get("data_source")},
        )
