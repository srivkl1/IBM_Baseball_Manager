"""Shared Streamlit UI components."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from backend.workflow import AgentResponse
from .theme import PALETTE


def agent_trace(resp: AgentResponse):
    with st.expander("🧠 Agent trace (orchestrator → retrieval → analysis → explanation)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Orchestrator plan**")
            st.json({
                "intent": resp.plan.intent,
                "confidence": round(resp.plan.confidence, 2),
                "data_request": resp.plan.data_request,
            })
            st.markdown("**Data retrieval summary**")
            st.json(resp.data_bundle_summary)
        with c2:
            st.markdown("**Analysis recommendation**")
            st.json({
                "headline": resp.recommendation.headline,
                "candidates": resp.recommendation.candidates[:5],
                "metrics": resp.recommendation.metrics,
            })
            st.markdown("**Self-evaluation**")
            st.json(resp.self_eval)
        st.caption(f"LLM provider: `{resp.trace.get('provider')}` • "
                   f"Data source: `{resp.trace.get('data_source')}`")


def recommendation_card(resp: AgentResponse):
    st.markdown(
        f"<div class='wb-card'>"
        f"<div><span class='wb-badge'>{resp.plan.intent}</span>"
        f"<span class='wb-badge'>{resp.trace.get('provider')} LLM</span></div>"
        f"<h3 style='margin-top:8px;color:{PALETTE['field_green']};'>"
        f"{resp.recommendation.headline}</h3>"
        f"<div>{resp.explanation}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def provider_pill():
    import os
    provider = os.getenv("LLM_PROVIDER", "mock")
    color = {"watsonx": PALETTE["home_red"], "custom": PALETTE["field_green"],
             "mock": PALETTE["leather_brown"]}.get(provider, PALETTE["away_navy"])
    st.markdown(
        f"<div style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color};color:white;font-size:0.8rem;'>"
        f"LLM: {provider}</div>",
        unsafe_allow_html=True,
    )
