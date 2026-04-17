# ⚾ Werbley's Squad — Fantasy Baseball Optimizer

End-to-end agentic AI MVP for the IBM Experiential AI Learning Lab.

A 4-agent pipeline (Orchestrator → Data Retrieval → Analysis → Explanation)
that powers an interactive **snake-draft simulator**, a **season tracker**
with historical 2025 replay, and a pluggable LLM backend (IBM watsonx.ai,
any OpenAI-compatible endpoint, or a deterministic mock for offline demos).

---

## 1 · Quick start

```bash
git clone <this-repo>
cd WERBLEYS_SQUAD
./run.sh          # creates .venv, installs deps, launches Streamlit
```

Open <http://localhost:8501>.

> First launch works 100% offline thanks to the synthetic fallback dataset.
> To use **real** pybaseball + ESPN data, fill in `.env` (see below).

## 2 · Environment variables

Copy `.env.example` → `.env` and edit:

```bash
# Pick one: "watsonx" | "custom" | "mock"
LLM_PROVIDER=mock

# IBM watsonx.ai — use your IBM Cloud credentials.
WATSONX_APIKEY=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=...
WATSONX_MODEL_ID=ibm/granite-3-8b-instruct

# Any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, LM Studio, …).
CUSTOM_LLM_BASE_URL=http://localhost:8080/v1
CUSTOM_LLM_API_KEY=not-needed
CUSTOM_LLM_MODEL=llama-3.1-8b-instruct

# Optional ESPN private-league credentials.
ESPN_LEAGUE_ID=
ESPN_SEASON=2025
ESPN_SWID=
ESPN_S2=
```

## 3 · Architecture

```
┌──────────────┐   intent + data request   ┌────────────────┐
│ Agent 1      │ ─────────────────────────▶│ Agent 2        │
│ Orchestrator │                            │ Data Retrieval │
└─────┬────────┘                            └──────┬─────────┘
      ▲                                           │ pybaseball + espn-api
      │ explanation                               ▼
┌─────┴────────┐    recommendation    ┌───────────────────┐
│ Agent 4      │ ◀────────────────────│ Agent 3           │
│ Explanation  │                       │ Analysis + Draft │
└──────────────┘                       └───────────────────┘
```

* **Agent 1 — Orchestrator** (`backend/agents/orchestrator.py`) classifies the
  user intent (`draft_pick`, `roster_move`, `player_trend`, `standings_check`)
  using a rules-first pass, falling back to the LLM for low-confidence cases,
  and emits a typed data request.
* **Agent 2 — Data Retrieval** (`backend/agents/data_retrieval.py`) pulls
  three-year FanGraphs + Statcast player data via `pybaseball`, FanGraphs top
  prospects, and ESPN league state via `espn-api`. Everything is disk-cached.
* **Agent 3 — Analysis** (`backend/agents/analysis.py`) calls the draft
  simulator (`backend/draft/simulator.py`) and the ML draft optimizer
  (`backend/models/draft_optimizer.py`, Gradient Boosted Regressor trained on
  2023 → 2024 transitions and evaluated OOT on 2024 → 2025) to rank candidates
  for the team on the clock, weighted by roster needs.
* **Agent 4 — Explanation** (`backend/agents/explanation.py`) rewrites the
  structured recommendation in plain English, tone calibrated to
  `beginner`/`expert`, then self-evaluates the output.

The full pipeline lives in `backend/workflow.py` and returns an
`AgentResponse` with every intermediate artifact, which the UI renders in an
expandable "agent trace" panel.

## 4 · UI pages

* **Home** — free-form chat-style questions, full trace view.
* **Draft Room** — live snake draft; CPU opponents pick with a BPA +
  positional-need strategy and the agent suggests your pick each round.
* **Season Tracker** — pick an as-of date (top-right) anywhere in the
  2025 MLB season and watch standings evolve; historical replay lets you
  stress-test your draft.
* **Model Lab** — retrain and inspect the ML optimizer's OOT metrics.

## 5 · Data scope

Per spec:
* **3-year scope**: 2023, 2024, 2025.
* **Train**: 2023 + 2024 (`prior-season features → next-season fantasy pts`).
* **OOT**: 2024 → 2025.
* Player pool is built from `pybaseball.batting_stats()` +
  `pybaseball.pitching_stats()` with FanGraphs advanced metrics (wRC+, wOBA,
  ISO, Barrel%, HardHit%, FIP, xFIP, K%, BB%, WAR, …).

## 6 · Switching LLM providers

```python
# backend/llm/base.py → get_llm()
LLM_PROVIDER=watsonx  # IBM Cloud granite / llama / mixtral models
LLM_PROVIDER=custom   # Any OpenAI-compatible HTTP endpoint
LLM_PROVIDER=mock     # Offline demo mode
```

All three providers implement the same `LLM.generate(prompt, system, …)`
signature so the agent code is provider-agnostic.

## 7 · Project layout

```
WERBLEYS_SQUAD/
├── app.py                       # Streamlit entry point
├── run.sh                       # venv + deps + launch
├── requirements.txt
├── .env.example
├── backend/
│   ├── config.py
│   ├── workflow.py              # end-to-end pipeline
│   ├── llm/                     # watsonx / custom / mock
│   ├── agents/                  # Agent 1..4
│   ├── data/                    # pybaseball + espn-api clients + cache
│   ├── draft/                   # simulator, player pool, scorer
│   └── models/                  # ML draft optimizer
└── frontend/
    ├── theme.py                 # baseball color palette + CSS
    ├── components.py
    └── pages/                   # home, draft, season_tracker, model_lab
```
