#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer Python 3.13/3.12 if available (pybaseball still pins pandas<2.2
# which has no wheels for Python 3.14). On Windows Git Bash, Chocolatey shims
# can exist even when the underlying Python is missing, so we verify candidates
# by executing `--version` and fall back to the Windows `py` launcher.
pick_python() {
  if command -v py >/dev/null 2>&1; then
    for cand in "-3.13" "-3.12" "-3.11" "-3.10" ""; do
      if py $cand --version >/dev/null 2>&1; then
        echo "py $cand"
        return
      fi
    done
  fi

  for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" --version >/dev/null 2>&1; then
      echo "$cand"
      return
    fi
  done

  echo ""
}
PY="$(pick_python)"

if [ -z "$PY" ]; then
  echo "No working Python interpreter found."
  echo "Install Python 3.12 or 3.11, then rerun this script."
  exit 1
fi

read -r -a PY_CMD <<< "$PY"
echo "Using interpreter: $("${PY_CMD[@]}" --version) ($PY)"

if [ ! -d .venv ]; then
  "${PY_CMD[@]}" -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  echo "Could not find a virtualenv activation script in .venv."
  exit 1
fi
<<<<<<< HEAD
=======

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  echo "[error] Could not find a virtualenv activation script in .venv."
  exit 1
fi
>>>>>>> 88c94826afd8a8ad2ebed89a449ebe7d21593e76

python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Optional real-data sources — skip silently if unavailable on this Python.
pip install --quiet -r requirements-optional.txt 2>/dev/null \
  || echo "[warn] pybaseball/espn-api did not install. App will use synthetic data."

# Optional LLM SDKs — skip silently if the user is running in mock mode.
pip install --quiet -r requirements-llm.txt 2>/dev/null || true

[ -f .env ] || cp .env.example .env
exec streamlit run app.py
