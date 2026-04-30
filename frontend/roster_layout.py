"""Shared ESPN-style roster ordering helpers."""
from __future__ import annotations

import pandas as pd


BATTER_SLOT_ORDER = ["C", "1B", "2B", "SS", "3B", "OF", "UTIL", "DH", "IF", "2B/SS", "1B/3B"]
PITCHER_SLOT_ORDER = ["SP", "RP", "P"]
PITCHER_POSITIONS = {"SP", "RP", "P"}


def _text(row: pd.Series, *keys: str) -> str:
    for key in keys:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip().upper()
    return ""


def _is_pitcher(row: pd.Series) -> bool:
    role = _text(row, "role")
    slot = _text(row, "lineup_slot", "Slot")
    pos = _text(row, "position", "pos", "fantasy_position")
    eligible = _text(row, "eligible")
    if role == "PIT":
        return True
    if slot in PITCHER_POSITIONS or pos in PITCHER_POSITIONS:
        return True
    return any(part.strip() in PITCHER_POSITIONS for part in eligible.replace("/", ",").split(","))


def _slot(row: pd.Series) -> str:
    slot = _text(row, "lineup_slot", "Slot")
    if slot:
        return slot
    return _text(row, "position", "pos", "fantasy_position", "role") or "BE"


def _slot_rank(slot: str, pitcher: bool) -> int:
    order = PITCHER_SLOT_ORDER if pitcher else BATTER_SLOT_ORDER
    if slot in order:
        return order.index(slot)
    if slot in {"LF", "CF", "RF"} and "OF" in order:
        return order.index("OF")
    return len(order)


def add_roster_layout(df: pd.DataFrame) -> pd.DataFrame:
    """Add ESPN-like slot/section ordering while preserving original columns."""
    if df.empty:
        return df.copy()

    ordered = df.copy()
    rows = []
    for idx, row in ordered.iterrows():
        slot = _slot(row)
        pitcher = _is_pitcher(row)
        is_il = slot == "IL"
        is_bench = slot == "BE"

        if pitcher:
            group = 5 if is_il else 4 if is_bench else 3
            section = "Pitcher IL" if is_il else "Pitcher bench" if is_bench else "Pitchers"
        else:
            group = 2 if is_il else 1 if is_bench else 0
            section = "Batter IL" if is_il else "Batter bench" if is_bench else "Batters"

        display_slot = slot
        if slot in {"LF", "CF", "RF"}:
            display_slot = "OF"
        elif slot in {"BE", "IL"}:
            display_slot = slot

        rows.append({
            "_roster_index": idx,
            "Roster area": section,
            "Slot": display_slot,
            "_roster_group": group,
            "_slot_rank": _slot_rank(display_slot, pitcher),
            "_player_rank": str(row.get("Name", row.get("Player", row.get("player", "")))),
        })

    layout = pd.DataFrame(rows).set_index("_roster_index")
    ordered = ordered.join(layout)
    return (
        ordered.sort_values(["_roster_group", "_slot_rank", "_player_rank"])
        .drop(columns=["_roster_group", "_slot_rank", "_player_rank"])
        .reset_index(drop=True)
    )
