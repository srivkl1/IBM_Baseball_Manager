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

    st.session_state.setdefault("home_last_user_text", "")
    st.session_state.setdefault("home_last_response", None)

    skill = st.radio("Skill level", ["beginner", "expert"], horizontal=True)

    # Quick-launch buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Who should I pick next?"):
            st.session_state["home_user_text"] = "Who should I pick next?"
    with col2:
        if st.button("Should I trade X for Y?"):
            st.session_state["home_user_text"] = "Should I trade X for Y?"
    with col3:
        if st.button("Is [player] trending up?"):
            st.session_state["home_user_text"] = "Is [player] trending up?"

    with st.form("home_chat_form"):
        user_text = st.text_input(
            "What do you want advice on?",
            placeholder="e.g., 'Who should I pick next?' or 'Is Elly De La Cruz trending up?'",
            key="home_user_text",
        )
        submit = st.form_submit_button("Ask Werbley")

    if submit and user_text.strip():
        with st.spinner("Werbley is thinking…"):
            resp = ensure_pipeline().run(
                user_text=user_text, skill_level=skill,
                draft_state=st.session_state.get("draft_state"),
                standings_table=st.session_state.get("standings_table"),
            )
        st.session_state["home_last_user_text"] = user_text
        st.session_state["home_last_response"] = resp

    if st.session_state["home_last_response"] is not None:
        st.markdown(f"**Last question:** {st.session_state['home_last_user_text']}")
        recommendation_card(st.session_state["home_last_response"])
        agent_trace(st.session_state["home_last_response"])
