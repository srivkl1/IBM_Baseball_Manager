"""Show the ML optimizer metrics for the configured historical training range."""
from __future__ import annotations

import streamlit as st

from backend.models.draft_optimizer import train_and_evaluate
from frontend.components import loading_state
from frontend.theme import page_header


def render():
    page_header("Model Lab", "Draft optimizer trained on historical year-to-year transitions.")

    if st.button("(Re)train optimizer"):
        with loading_state(
            "Training optimizer",
            "Rebuilding hitter and pitcher models from historical season pairs.",
        ):
            metrics = train_and_evaluate(force=True)
        st.success("Retrained.")
    else:
        with loading_state(
            "Loading model metrics",
            "Checking cached optimizer results and out-of-time validation.",
        ):
            metrics = train_and_evaluate()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hitter model")
        st.metric("OOT MAE (fantasy pts)", f"{metrics.mae_bat_oot:.1f}")
        st.metric("OOT R²", f"{metrics.r2_bat_oot:.2f}")
    with c2:
        st.subheader("Pitcher model")
        st.metric("OOT MAE (fantasy pts)", f"{metrics.mae_pit_oot:.1f}")
        st.metric("OOT R²", f"{metrics.r2_pit_oot:.2f}")

    st.caption(
        f"Training range: {metrics.train_range[0]}-{metrics.train_range[1]} "
        f"across {metrics.train_pairs} year-pairs - OOT: {metrics.oot_pair[0]}->{metrics.oot_pair[1]}"
    )
