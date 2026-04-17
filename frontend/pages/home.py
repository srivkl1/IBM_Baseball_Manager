"""Home / chat-style interaction with the 4-agent pipeline."""
from __future__ import annotations

import streamlit as st

from backend.workflow import AgentPipeline
from frontend.components import agent_trace, recommendation_card
from frontend.theme import page_header


def render():
    page_header("Werbley's Squad — Fantasy Baseball Assistant",
                "Ask for draft picks, trade ideas, player trends, or standings.")

    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = AgentPipeline()

    skill = st.radio("Skill level", ["beginner", "expert"], horizontal=True)

    user_text = st.text_input(
        "What do you want advice on?",
        placeholder="e.g., 'Who should I pick next?' or 'Is Elly De La Cruz trending up?'",
    )
    if st.button("Ask the squad") and user_text.strip():
        with st.spinner("Agents working the count…"):
            resp = st.session_state["pipeline"].run(
                user_text=user_text, skill_level=skill,
                draft_state=st.session_state.get("draft_state"),
                standings_table=st.session_state.get("standings_table"),
            )
        recommendation_card(resp)
        agent_trace(resp)
