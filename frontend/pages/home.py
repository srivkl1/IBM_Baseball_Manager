"""Home / chat-style interaction with the 4-agent pipeline."""
from __future__ import annotations

import streamlit as st

from backend.draft import league_state
from frontend.components import agent_trace, ensure_pipeline, loading_state, recommendation_card
from frontend.theme import page_header


def render():
    page_header(
        "Werbley's Squad - Fantasy Baseball Assistant",
        "Ask for draft picks, trade ideas, player trends, standings, or baseball facts.",
    )

    ensure_pipeline()

    if "draft_state" not in st.session_state:
        imported_state, bundle, _ = league_state.load_existing_league_state()
        if imported_state is not None:
            st.session_state["draft_state"] = imported_state
            st.session_state["draft_bundle"] = bundle

    st.session_state.setdefault("home_last_user_text", "")
    st.session_state.setdefault("home_last_response", None)

    skill = st.radio("Skill level", ["beginner", "expert"], horizontal=True)

    st.caption("Try a prompt")
    prompt_rows = [
        [
            ("Top draft picks", "Who are the top draft picks for this year's draft?"),
            ("Top OF", "Who are the top outfielders for this year's draft?"),
            ("Top SP", "Who are the top starting pitchers?"),
        ],
        [
            ("Best catchers", "List the best catchers for fantasy baseball."),
            ("Trade idea", "Should I trade X for Y?"),
            ("Player trend", "Why is Roki Sasaki performing poorly?"),
        ],
    ]
    for row in prompt_rows:
        cols = st.columns(len(row))
        for col, (label, prompt) in zip(cols, row):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state["home_user_text"] = prompt

    with st.form("home_chat_form"):
        user_text = st.text_input(
            "What do you want advice on?",
            placeholder="e.g., 'Who should I pick next?' or 'Why is Roki Sasaki performing poorly?'",
            key="home_user_text",
        )
        submit = st.form_submit_button("Ask Werbley")

    if submit and user_text.strip():
        with loading_state(
            "Werbley is checking the matchup board",
            "Pulling fantasy context, MLB data, and recent-stat signals.",
        ):
            resp = ensure_pipeline().run(
                user_text=user_text,
                skill_level=skill,
                draft_state=st.session_state.get("draft_state"),
                standings_table=st.session_state.get("standings_table"),
            )
        st.session_state["home_last_user_text"] = user_text
        st.session_state["home_last_response"] = resp

    if st.session_state["home_last_response"] is not None:
        st.markdown(f"**Last question:** {st.session_state['home_last_user_text']}")
        recommendation_card(st.session_state["home_last_response"])
        agent_trace(st.session_state["home_last_response"])
