"""Home / chat-style interaction with the 4-agent pipeline."""
from __future__ import annotations

import streamlit as st

from backend.draft import league_state
from frontend.components import agent_trace, ensure_pipeline, recommendation_card
from frontend.theme import page_header


def render():
    page_header("Werbley's Squad — Fantasy Baseball Assistant",
                "Ask for draft picks, trade ideas, player trends, or standings.")

    ensure_pipeline()

    if "draft_state" not in st.session_state:
        imported_state, bundle, _ = league_state.load_existing_league_state()
        if imported_state is not None:
            st.session_state["draft_state"] = imported_state
            st.session_state["draft_bundle"] = bundle

    skill = st.radio("Skill level", ["beginner", "expert"], horizontal=True)

    user_text = st.text_input(
        "What do you want advice on?",
        placeholder="e.g., 'Who should I pick next?' or 'Is Elly De La Cruz trending up?'",
    )
    if st.button("Ask the squad") and user_text.strip():
        with st.spinner("Werbley is thinking…"):
            resp = ensure_pipeline().run(
                user_text=user_text, skill_level=skill,
                draft_state=st.session_state.get("draft_state"),
                standings_table=st.session_state.get("standings_table"),
            )
        recommendation_card(resp)
        agent_trace(resp)
