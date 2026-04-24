#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer Python 3.13/3.12 if available (pybaseball still pins pandas<2.2
# which has no wheels for Python 3.14). Fall back to whatever `python3` is.
pick_python() {
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then echo "$cand"; return; fi
  done
  echo "python3"
}
PY="$(pick_python)"
echo "Using interpreter: $($PY --version) ($PY)"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  echo "[error] Could not find a virtualenv activation script in .venv."
  exit 1
fi

python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Optional real-data sources — skip silently if unavailable on this Python.
pip install --quiet -r requirements-optional.txt 2>/dev/null \
  || echo "[warn] pybaseball/espn-api did not install. App will use synthetic data."

# Optional LLM SDKs — skip silently if the user is running in mock mode.
pip install --quiet -r requirements-llm.txt 2>/dev/null || true

[ -f .env ] || cp .env.example .env
exec streamlit run app.py
