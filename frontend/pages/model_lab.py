"""Show the ML optimizer metrics (2023+2024 train / 2025 OOT)."""
from __future__ import annotations

import streamlit as st

from backend.models.draft_optimizer import train_and_evaluate
from frontend.theme import page_header


def render():
    page_header("Model Lab", "Draft optimizer — trained on 2023/24, evaluated on 2025 OOT.")

    if st.button("(Re)train optimizer"):
        metrics = train_and_evaluate(force=True)
        st.success("Retrained.")
    else:
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

    st.caption(f"Train seasons: {metrics.train_seasons} — OOT season: {metrics.oot_season}")
