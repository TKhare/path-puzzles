# path puzzles

Connect each pair of matching letters with a path, minimizing empty squares (paths may overlap).
A showcase of small puzzles that frontier models get wrong — play them yourself, watch a model
attempt one live, and see measured per-model pass rates (Opus 4.5 / Sonnet 4.6 / Opus 4.8 at
low·high·max effort) in the Stats tab.

## Deploy (Render)
This repo includes `render.yaml`. Create a free Render **Web Service** from this repo; Render reads
the blueprint automatically. Then set the secret env var **`ANTHROPIC_API_KEY`** in the dashboard
(the live "run a model" feature needs it). Guardrails: `LIVE_DAILY_CAP` (USD/day) and
`LIVE_PER_IP_HOUR` bound the live feature's API spend.

Run locally: `pip install -r requirements.txt && gunicorn --chdir webapp app:app` (or
`python webapp/app.py`), then open the printed URL.
