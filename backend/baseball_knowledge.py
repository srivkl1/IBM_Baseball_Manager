"""Stable baseball facts for general Q&A fallback.

This is intentionally small and conservative. Live/current player facts should
come from MLB/ESPN APIs; this module handles durable baseball rules, stat
definitions, and MLB team/city questions when the LLM is unavailable or too
rigid.
"""
from __future__ import annotations

import re
from typing import Optional


MLB_TEAMS = [
    {"team": "Arizona Diamondbacks", "city": "Phoenix", "state": "Arizona", "league": "NL", "division": "West"},
    {"team": "Atlanta Braves", "city": "Atlanta", "state": "Georgia", "league": "NL", "division": "East"},
    {"team": "Baltimore Orioles", "city": "Baltimore", "state": "Maryland", "league": "AL", "division": "East"},
    {"team": "Boston Red Sox", "city": "Boston", "state": "Massachusetts", "league": "AL", "division": "East"},
    {"team": "Chicago Cubs", "city": "Chicago", "state": "Illinois", "league": "NL", "division": "Central"},
    {"team": "Chicago White Sox", "city": "Chicago", "state": "Illinois", "league": "AL", "division": "Central"},
    {"team": "Cincinnati Reds", "city": "Cincinnati", "state": "Ohio", "league": "NL", "division": "Central"},
    {"team": "Cleveland Guardians", "city": "Cleveland", "state": "Ohio", "league": "AL", "division": "Central"},
    {"team": "Colorado Rockies", "city": "Denver", "state": "Colorado", "league": "NL", "division": "West"},
    {"team": "Detroit Tigers", "city": "Detroit", "state": "Michigan", "league": "AL", "division": "Central"},
    {"team": "Houston Astros", "city": "Houston", "state": "Texas", "league": "AL", "division": "West"},
    {"team": "Kansas City Royals", "city": "Kansas City", "state": "Missouri", "league": "AL", "division": "Central"},
    {"team": "Los Angeles Angels", "city": "Anaheim", "state": "California", "league": "AL", "division": "West"},
    {"team": "Los Angeles Dodgers", "city": "Los Angeles", "state": "California", "league": "NL", "division": "West"},
    {"team": "Miami Marlins", "city": "Miami", "state": "Florida", "league": "NL", "division": "East"},
    {"team": "Milwaukee Brewers", "city": "Milwaukee", "state": "Wisconsin", "league": "NL", "division": "Central"},
    {"team": "Minnesota Twins", "city": "Minneapolis", "state": "Minnesota", "league": "AL", "division": "Central"},
    {"team": "New York Mets", "city": "New York", "state": "New York", "league": "NL", "division": "East"},
    {"team": "New York Yankees", "city": "New York", "state": "New York", "league": "AL", "division": "East"},
    {"team": "Oakland Athletics", "city": "Oakland", "state": "California", "league": "AL", "division": "West"},
    {"team": "Philadelphia Phillies", "city": "Philadelphia", "state": "Pennsylvania", "league": "NL", "division": "East"},
    {"team": "Pittsburgh Pirates", "city": "Pittsburgh", "state": "Pennsylvania", "league": "NL", "division": "Central"},
    {"team": "San Diego Padres", "city": "San Diego", "state": "California", "league": "NL", "division": "West"},
    {"team": "San Francisco Giants", "city": "San Francisco", "state": "California", "league": "NL", "division": "West"},
    {"team": "Seattle Mariners", "city": "Seattle", "state": "Washington", "league": "AL", "division": "West"},
    {"team": "St. Louis Cardinals", "city": "St. Louis", "state": "Missouri", "league": "NL", "division": "Central"},
    {"team": "Tampa Bay Rays", "city": "Tampa Bay", "state": "Florida", "league": "AL", "division": "East"},
    {"team": "Texas Rangers", "city": "Arlington", "state": "Texas", "league": "AL", "division": "West"},
    {"team": "Toronto Blue Jays", "city": "Toronto", "state": "Ontario", "league": "AL", "division": "East"},
    {"team": "Washington Nationals", "city": "Washington", "state": "District of Columbia", "league": "NL", "division": "East"},
]


ALIASES = {
    "new york": "New York",
    "ny": "New York",
    "nyc": "New York",
    "la": "Los Angeles",
    "los angeles": "Los Angeles",
    "chicago": "Chicago",
    "florida": "Florida",
    "texas": "Texas",
    "california": "California",
    "ohio": "Ohio",
    "pennsylvania": "Pennsylvania",
    "missouri": "Missouri",
}


STAT_DEFINITIONS = {
    "ops": "OPS means on-base plus slugging. It combines how often a hitter reaches base with how much power they hit for.",
    "era": "ERA means earned run average. It estimates how many earned runs a pitcher allows per nine innings.",
    "whip": "WHIP means walks plus hits per inning pitched. Lower is better because it means fewer baserunners allowed.",
    "rbi": "RBI means runs batted in. A hitter gets an RBI when their plate appearance drives in a run, with a few scoring-rule exceptions.",
    "war": "WAR means wins above replacement. It estimates a player's total value compared with a readily available replacement player.",
    "wrc+": "wRC+ measures a hitter's total offensive production adjusted for park and league. 100 is league average; above 100 is better.",
    "fip": "FIP estimates pitching performance from strikeouts, walks, hit batters, and home runs, removing much of the defense/noise behind ERA.",
    "xfip": "xFIP is like FIP, but it normalizes home-run rate to estimate what a pitcher's ERA indicators may look like going forward.",
    "babip": "BABIP means batting average on balls in play. It can hint at luck, defense, contact quality, and speed.",
    "obp": "OBP means on-base percentage. It measures how often a hitter reaches base by hit, walk, or hit-by-pitch.",
    "slg": "SLG means slugging percentage. It measures power by weighting extra-base hits more than singles.",
    "save": "A save is credited to a reliever who finishes a win while protecting a qualifying lead under MLB scoring rules.",
    "hold": "A hold is credited to a reliever who enters in a save situation, preserves the lead, and leaves before the game ends.",
}


RULE_ANSWERS = {
    "innings": "A regulation MLB game is scheduled for 9 innings. If it is tied after 9, it goes to extra innings.",
    "outs": "Each half-inning has 3 outs. A full inning has 6 outs total, 3 for each team.",
    "strikeout": "A strikeout happens when a batter records three strikes during a plate appearance.",
    "walk": "A walk happens when a pitcher throws four balls to a batter, sending the batter to first base.",
    "dh": "DH means designated hitter. The DH bats in place of the pitcher or another listed defensive player, depending on league rules.",
    "bullpen": "The bullpen is the group of relief pitchers and the area where they warm up during a game.",
    "closer": "A closer is usually the reliever a team uses to finish close games, especially save chances.",
}


def answer_basic_question(question: str) -> Optional[str]:
    q = re.sub(r"[?!.]", "", question or "").strip().lower()
    if not q:
        return None

    team_answer = _answer_team_location(q)
    if team_answer:
        return team_answer

    stat_answer = _answer_stat_definition(q)
    if stat_answer:
        return stat_answer

    if "how many" in q and "team" in q and ("mlb" in q or "baseball" in q):
        return "MLB has 30 teams: 15 in the American League and 15 in the National League."

    if "american league" in q or re.search(r"\bal\b", q):
        return "The American League is one of MLB's two leagues. It has 15 teams split across East, Central, and West divisions."
    if "national league" in q or re.search(r"\bnl\b", q):
        return "The National League is one of MLB's two leagues. It has 15 teams split across East, Central, and West divisions."

    for key, answer in RULE_ANSWERS.items():
        if key in q and any(word in q for word in ("what", "how", "explain", "mean", "define")):
            return answer
    return None


def _answer_team_location(q: str) -> Optional[str]:
    if not any(word in q for word in ("team", "teams", "mlb", "baseball")):
        return None
    location = None
    for alias, canonical in ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            location = canonical
            break
    if not location:
        return None

    matches = [
        row for row in MLB_TEAMS
        if row["city"].casefold() == location.casefold()
        or row["state"].casefold() == location.casefold()
    ]
    if not matches:
        return None
    names = ", ".join(row["team"] for row in matches)
    if len(matches) == 1:
        row = matches[0]
        return f"The MLB team from {location} is the {row['team']} ({row['league']} {row['division']})."
    return f"The MLB teams from {location} are {names}."


def _answer_stat_definition(q: str) -> Optional[str]:
    if not any(word in q for word in ("what", "mean", "means", "define", "explain", "is")):
        return None
    for stat, answer in STAT_DEFINITIONS.items():
        if re.search(rf"\b{re.escape(stat)}\b", q):
            return answer
    return None
