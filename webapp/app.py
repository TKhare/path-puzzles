"""Flask backend for the path-puzzle showcase webapp.

Serves the hand-found "hard" puzzles, lets a user play them, and streams a
chosen model's attempt live (its prompt + thinking trace), then runs a cheap
"converter" model that reads the trace and emits the answer in a standardized
path format so the attempt can be drawn on the grid and graded. Per-puzzle /
per-model pass rates are persisted to stats.json and updated on every attempt.

Run:  .venv/bin/python webapp/app.py     (then open http://127.0.0.1:5050)
"""
import datetime
import json
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import (parse_board, board_to_text, letter_cells, make_prompt,
                  parse_final_answer, check_answer, render_with_paths)

import anthropic

# ---- load ANTHROPIC_API_KEY from project .env --------------------------------
if "ANTHROPIC_API_KEY" not in os.environ and (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
        if m:
            os.environ["ANTHROPIC_API_KEY"] = m.group(1)

client = anthropic.Anthropic(max_retries=2, timeout=600.0)

WEB = Path(__file__).resolve().parent
STATS_FILE = WEB / "stats.json"
STATS_LOCK = threading.Lock()
SOLVES_FILE = WEB / "solves.json"   # per-puzzle human solve times (first touch -> solve)
SOLVES_LOCK = threading.Lock()

# ---- public-deploy guardrails for the live "run a model" feature -------------
# Each live run spends real API credits, so cap total daily spend and per-visitor rate.
LIVE_DAILY_CAP = float(os.environ.get("LIVE_DAILY_CAP", "3.0"))   # USD/day across all visitors
LIVE_PER_IP_HOUR = int(os.environ.get("LIVE_PER_IP_HOUR", "6"))   # runs per IP per rolling hour
_live_lock = threading.Lock()
_live = {"day": None, "spend": 0.0, "ip": defaultdict(list)}


def live_gate(ip):
    """Return None if a live run is allowed for `ip`, else a user-facing reason string."""
    with _live_lock:
        today = datetime.date.today().isoformat()
        if _live["day"] != today:
            _live.update(day=today, spend=0.0, ip=defaultdict(list))
        now = time.time()
        recent = [t for t in _live["ip"][ip] if now - t < 3600]
        _live["ip"][ip] = recent
        if _live["spend"] >= LIVE_DAILY_CAP:
            return "Daily demo budget reached — live runs resume tomorrow. The Stats tab shows pre-measured results."
        if len(recent) >= LIVE_PER_IP_HOUR:
            return f"Rate limit: up to {LIVE_PER_IP_HOUR} live runs per hour. Try again later, or see the Stats tab."
        recent.append(now)
        return None


def live_spent(out_tok):
    with _live_lock:
        _live["spend"] += out_tok * 25 / 1e6   # conservative (Opus output rate) accounting

# ---- the showcase puzzles (deploy set), strongest result first ---------------
# Inclusion criteria: a puzzle is shown ONLY if Opus 4.8 has a documented genuine
# failure on it — either (A) it FAILS even at MAX effort (stops voluntarily at
# stop=end_turn, budget to spare, and submits a sub-optimal answer with a false
# optimality claim), or (B) it fails CONSISTENTLY at HIGH effort.
#   p00..p02 — MAX-effort resignations: hov_01 (~50%), s1_09 (~40%), subtle_05 (~33%).
#   p03..p05 — consistent HIGH-effort failures (solved at max): p00, s1_01, s1_02.
HARD_ORDER = ["hov_01", "s1_09", "subtle_05", "p00", "s1_01", "s1_02"]
# hide from the stats table: historical, plus the bare live labels (no-thinking) that aren't
# part of the documented 5-config matrix (Opus 4.5 / Sonnet 4.6 / Opus 4.8 low|high|max).
HIDE_MODELS = {"Fable 5", "Opus 4.8", "Haiku 4.5"}
# Public display ids: the site shows p00..pNN regardless of internal generation ids.
DISPLAY = {pid: f"p{i:02d}" for i, pid in enumerate(HARD_ORDER)}

ALL_PUZ = {}
for f in ("puzzles.json", "puzzles2.json", "puzzles3.json",
          "puzzles4.json", "puzzles5.json", "puzzles6.json", "puzzles7.json",
          "puzzles8.json"):
    if (ROOT / f).exists():
        for p in json.load(open(ROOT / f)):
            ALL_PUZ[p["id"]] = p


def puzzle_payload(pid):
    p = ALL_PUZ[pid]
    grid = parse_board(p["board"])
    lc = letter_cells(grid)
    letters = {ch: [[r + 1, c + 1] for r, c in cells] for ch, cells in lc.items()}
    return {
        "id": pid, "board": p["board"], "rows": p["rows"], "cols": p["cols"],
        "optimal": p["optimal"], "n_letters": len(lc), "letters": letters,
        "solution": p.get("solution"), "prompt": make_prompt(grid) + TOOL_DISPLAY,
        "display": DISPLAY.get(pid, pid),
    }


# ---- runnable solver models --------------------------------------------------
# NOTE: no extended-thinking param on purpose. With thinking on, the reasoning is
# hidden in (summarized/omitted) thinking blocks; with it off, the model reasons
# out loud in plain text — which is what we stream and what the watcher reads.
RUNNABLE = {
    "opus-4-8":   {"label": "Opus 4.8",   "id": "claude-opus-4-8",   "max_tokens": 22000},
    "sonnet-4-6": {"label": "Sonnet 4.6", "id": "claude-sonnet-4-6", "max_tokens": 22000},
    "haiku-4-5":  {"label": "Haiku 4.5",  "id": "claude-haiku-4-5",  "max_tokens": 16000},
}

# Haiku watcher: (a) infers the live board from the solver's streaming reasoning BETWEEN
# see_board calls, and (b) extracts the FINAL answer (convert). see_board calls push
# ground-truth board states on top of the watcher's guesses.
WATCHER_PREFERRED = "claude-haiku-4-7"
WATCHER_FALLBACK = "claude-haiku-4-5"
_watcher_id = {"v": None}
WATCH_INTERVAL = 1.6     # seconds between watcher snapshots (fills gaps between see_board calls)
WATCH_MIN_NEW = 450      # min new chars of reasoning before re-snapshotting


def watcher_complete(prompt, max_tokens=1200):
    ids = [_watcher_id["v"]] if _watcher_id["v"] else [WATCHER_PREFERRED, WATCHER_FALLBACK]
    last = None
    for mid in ids:
        try:
            msg = client.messages.create(model=mid, max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
            _watcher_id["v"] = mid
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:
            last = e
    raise last


# see_board: the solver can render its proposed paths (a visual-feedback channel). Each
# legal call also pushes that exact board to the live view. Prompt caching keeps the
# growing tool-loop transcript cheap (re-sent prefix bills at the cache-read rate).
SEE_BOARD_TOOL = {
    "name": "see_board",
    "description": (
        "Render the board with your proposed paths so you can see your current attempt. "
        "`paths` maps each letter to its path: a list of [row,col] pairs (1-indexed) from one "
        "copy of the letter to the other; include any subset of letters. Endpoints show as the "
        "UPPERCASE letter, squares a path passes through show as that letter in lowercase, empty "
        "squares as '_'. On overlap, the square shows whichever letter is listed FIRST in your "
        "call. If a path is illegal (wrong endpoints, a non-adjacent step, crosses itself, or "
        "passes through another letter) the tool returns INVALID with the reason. It also reports "
        "how many squares are still empty."),
    "input_schema": {
        "type": "object",
        "properties": {"paths": {"type": "object",
                                 "description": "letter -> list of [row,col] (1-indexed)"}},
        "required": ["paths"],
    },
}
# Shown in the "prompt sent to the model" panel so the visibility is complete — the tool
# is delivered via the API tools parameter, not appended to the user message text.
TOOL_DISPLAY = ("\n\n──── tool available to the model (provided via the API, not the text above) ────\n"
                "see_board(paths): " + SEE_BOARD_TOOL["description"])
MAX_ITERS = 30        # see_board backstop (the model used <=18); finalize cleanly if ever hit


def _set_cache(msgs):
    """One trailing ephemeral cache breakpoint -> incremental caching of the tool loop."""
    for m in msgs:
        if isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
    last = msgs[-1]["content"]
    if isinstance(last, list) and last and isinstance(last[-1], dict):
        last[-1]["cache_control"] = {"type": "ephemeral"}

# ---- stats persistence ------------------------------------------------------
def load_stats():
    if STATS_FILE.exists():
        return json.load(open(STATS_FILE))
    return {}


def update_stats(pid, model_label, correct, out_tok=None, secs=None):
    with STATS_LOCK:
        s = load_stats()
        cell = s.setdefault(pid, {}).setdefault(model_label, {"attempts": 0, "passes": 0})
        cell["attempts"] += 1
        cell["passes"] += 1 if correct else 0
        if out_tok is not None:                     # per-attempt perf (live runs only)
            cell["tok_sum"] = cell.get("tok_sum", 0) + int(out_tok)
            cell["sec_sum"] = round(cell.get("sec_sum", 0) + (secs or 0), 1)
            cell["perf_n"] = cell.get("perf_n", 0) + 1
        tmp = str(STATS_FILE) + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, STATS_FILE)
        return s[pid]


# ---- converter: trace -> standardized paths ---------------------------------
CONVERTER_PROMPT = """You extract a puzzle-solver's FINAL intended answer into strict JSON.

Grid (row 1 = top, column 1 = left):
{board}

Letters to connect: {letters}

Below is another AI's full attempt (its thinking, then its final output). Extract
the FINAL path it settled on for EACH letter: an ordered list of [row, col] cells
from one copy of the letter to the other. Prefer the model's explicit FINAL ANSWER
if present; otherwise infer its final intended paths from the reasoning. 1-indexed.

Output ONLY a JSON object, no prose, exactly:
{{"paths": {{"A": [[r,c],[r,c]], "B": [[r,c]]}}, "empty": <claimed empty count or null>}}

Attempt:
{transcript}
"""


def _json_paths(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
        raw = obj.get("paths", {})
        paths = {ch: [(int(r), int(c)) for r, c in cells]
                 for ch, cells in raw.items() if cells}
        return (paths or None), obj.get("empty")
    except Exception:
        return None, None


def convert(board_text, letters, answer):
    """Final extraction: read the whole reasoning, return its settled-on answer."""
    prompt = CONVERTER_PROMPT.format(board=board_text, letters=", ".join(letters),
                                     transcript=answer[-15000:])
    try:
        paths, claimed = _json_paths(watcher_complete(prompt, max_tokens=2000))
        if paths:
            return paths, claimed, "converter"
    except Exception:
        pass
    paths, claimed = parse_final_answer(answer)  # fallback: parse the model's own block
    return paths, claimed, "fallback"


LIVE_PROMPT = """Another AI is solving a path puzzle, thinking out loud. You are turning its
work-in-progress into a live drawing. Below is its reasoning so far — it may be mid-sentence
and full of false starts.

Grid (row 1 = top, column 1 = left):
{board}
Letters: {letters}

For each letter, output the most recent concrete route the AI is currently going with. Be
eager — if it has sketched any coordinates for a letter, draw them.

If the AI is in the act of TEARING UP a letter's route to try a different one (e.g. "wait,
that won't work — let me redo C") and has not yet settled on the new route, OMIT that letter
entirely so the board clears it, signalling that letter is being reworked. Add it back only
once the AI commits to its new route. Also omit any letter it has not touched yet.

Output ONLY JSON: {{"paths": {{"A": [[r,c],[r,c]], "B": [[r,c]]}}}}
Use {{"reset": true}} ONLY if it has not written any concrete cell coordinates at all yet.
Coordinates are 1-indexed (row, col). JSON only, no prose.

Reasoning so far:
{reasoning}
"""


def live_extract(board_text, letters, buf):
    """One watcher snapshot inferred from the streaming reasoning (or reset)."""
    prompt = LIVE_PROMPT.format(board=board_text, letters=", ".join(letters),
                                reasoning=buf[-10000:])
    try:
        txt = watcher_complete(prompt)
    except Exception:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if obj.get("reset"):
        return {"reset": True}
    raw = obj.get("paths", {})
    paths = {ch: [[int(r), int(c)] for r, c in cells] for ch, cells in raw.items() if cells}
    return {"paths": paths} if paths else {"reset": True}


# ---- SSE helpers ------------------------------------------------------------
def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


app = Flask(__name__)


@app.get("/")
def index():
    return send_file(WEB / "index.html")


@app.get("/api/puzzles")
def api_puzzles():
    order = [p for p in HARD_ORDER if p in ALL_PUZ]
    return jsonify({
        "puzzles": [puzzle_payload(p) for p in order],
        "runnable": [{"key": k, "label": v["label"]} for k, v in RUNNABLE.items()],
    })


@app.get("/api/stats")
def api_stats():
    stats = load_stats()
    order = [p for p in HARD_ORDER if p in ALL_PUZ]
    # the documented 5-config matrix, in display order
    MODEL_ORDER = ["Opus 4.5", "Sonnet 4.6", "Opus 4.8 (low)", "Opus 4.8 (high)", "Opus 4.8 (max)"]
    labels = []
    for k, v in RUNNABLE.items():
        if v["label"] not in labels and v["label"] not in HIDE_MODELS:
            labels.append(v["label"])
    for pid in order:
        for lab in stats.get(pid, {}):
            if lab not in labels and lab not in HIDE_MODELS:
                labels.append(lab)
    labels.sort(key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))
    return jsonify({
        "order": order,
        "models": labels,
        "runnable": [v["label"] for v in RUNNABLE.values()],
        "optima": {p: ALL_PUZ[p]["optimal"] for p in order},
        "sizes": {p: f'{ALL_PUZ[p]["rows"]}x{ALL_PUZ[p]["cols"]}' for p in order},
        "display": {p: DISPLAY.get(p, p) for p in order},
        "stats": stats,
    })


@app.get("/api/effort")
def api_effort():
    f = WEB / "effort_stats.json"
    if f.exists():
        return jsonify(json.load(open(f)))
    return jsonify({"curve": [], "max_per_puzzle": [], "trajectories": {}})


@app.post("/api/solve")
def api_solve():
    data = request.get_json(force=True, silent=True) or {}
    pid, secs = data.get("puzzle"), data.get("seconds")
    if pid not in ALL_PUZ or not isinstance(secs, (int, float)):
        return jsonify({"ok": False}), 400
    with SOLVES_LOCK:
        s = json.load(open(SOLVES_FILE)) if SOLVES_FILE.exists() else {}
        s.setdefault(pid, []).append(round(float(secs), 1))
        tmp = str(SOLVES_FILE) + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, SOLVES_FILE)
        n = len(s[pid])
    print(f"[solve] {pid} {secs:.1f}s ({n} total)", flush=True)
    return jsonify({"ok": True})


@app.get("/api/attempt")
def api_attempt():
    pid = request.args.get("puzzle")
    mkey = request.args.get("model")
    if pid not in ALL_PUZ or mkey not in RUNNABLE:
        return Response(sse("error", {"msg": "bad puzzle or model"}),
                        mimetype="text/event-stream")
    mcfg = RUNNABLE[mkey]
    p = ALL_PUZ[pid]
    grid = parse_board(p["board"])
    letters = sorted(letter_cells(grid).keys())
    prompt = make_prompt(grid)
    board_text = p["board"]
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr) or "?").split(",")[0].strip()
    blocked = live_gate(ip)   # rate-limit / daily-budget guard for the public deploy

    def gen():
        if blocked:
            yield sse("error", {"msg": blocked})
            return
        yield sse("prompt", {"text": prompt + TOOL_DISPLAY, "model": mcfg["label"]})
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        parts, tool_calls, out_tok, stop = [], 0, 0, None
        q = queue.Queue()
        st = {"buf": "", "done": False}
        lk = threading.Lock()

        def watcher():                        # infers the board from the streaming reasoning
            last = 0
            while True:
                time.sleep(WATCH_INTERVAL)
                with lk:
                    buf, done = st["buf"], st["done"]
                if buf.strip() and (done or len(buf) - last >= WATCH_MIN_NEW):
                    last = len(buf)
                    snap = live_extract(board_text, letters, buf)
                    if snap:
                        q.put(snap)
                if done:
                    break

        threading.Thread(target=watcher, daemon=True).start()
        t0 = time.time()
        try:
            for _ in range(MAX_ITERS + 4):
                _set_cache(msgs)
                kw = dict(model=mcfg["id"], max_tokens=mcfg["max_tokens"], messages=msgs)
                if tool_calls < MAX_ITERS:
                    kw["tools"] = [SEE_BOARD_TOOL]
                with client.messages.stream(**kw) as stream:
                    for event in stream:
                        if (event.type == "content_block_delta"
                                and getattr(event.delta, "type", "") == "text_delta"):
                            parts.append(event.delta.text)
                            with lk:
                                st["buf"] += event.delta.text
                            yield sse("text", {"t": event.delta.text})
                        while True:           # watcher (reasoning-inferred) snapshots
                            try:
                                yield sse("snapshot", q.get_nowait())
                            except queue.Empty:
                                break
                    final = stream.get_final_message()
                stop = final.stop_reason
                out_tok += final.usage.output_tokens
                if final.stop_reason != "tool_use" or tool_calls >= MAX_ITERS:
                    break
                msgs.append({"role": "assistant", "content": final.content})
                results = []
                for b in final.content:
                    if getattr(b, "type", "") != "tool_use":
                        continue
                    if b.name == "see_board":
                        tool_calls += 1
                        tpaths = (b.input or {}).get("paths", {})
                        text, err = render_with_paths(grid, tpaths)
                        if err is None:           # ground truth: drive the board to this exact state
                            snap = {ch: [[int(r), int(c)] for r, c in cells]
                                    for ch, cells in tpaths.items() if cells}
                            yield sse("snapshot", {"paths": snap})
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": text if err is None else f"INVALID path — {err}"})
                    else:
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": "unknown tool"})
                if tool_calls >= MAX_ITERS:
                    results.append({"type": "text", "text": "You have used your last see_board "
                                    "call. Now give your FINAL ANSWER in the exact required format."})
                msgs.append({"role": "user", "content": results})
        except Exception as e:
            with lk:
                st["done"] = True
            yield sse("error", {"msg": f"{type(e).__name__}: {e}"})
            return
        finally:
            with lk:                          # client disconnected or finished → stop the watcher
                st["done"] = True

        while True:                           # drain any final watcher snapshots
            try:
                yield sse("snapshot", q.get_nowait())
            except queue.Empty:
                break
        wall = round(time.time() - t0, 1)
        answer = "".join(parts)
        yield sse("status", {"s": "converting"})
        paths, claimed, source = convert(board_text, letters, answer)
        paths1 = {ch: [list(c) for c in cells] for ch, cells in paths.items()}
        graded = check_answer(grid, paths, p["optimal"])
        live_spent(out_tok)   # tally toward the daily cap; live runs do NOT persist to the curated stats
        yield sse("result", {
            "paths": paths1, "source": source, "claimed": claimed,
            "achieved": graded["achieved_blanks"], "optimal": p["optimal"],
            "valid": graded["valid"], "correct": graded["correct"], "errors": graded["errors"][:3],
            "stop_reason": stop, "output_tokens": out_tok, "seconds": wall, "tool_calls": tool_calls,
            "model": mcfg["label"], "puzzle": pid, "puzzle_stats": None,
        })
        yield sse("done", {})

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("Hard puzzles:", [p for p in HARD_ORDER if p in ALL_PUZ])
    app.run(host="127.0.0.1", port=5050, threaded=True, debug=False)
