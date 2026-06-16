"""Core logic for letter-path puzzles.

Puzzle: a grid where each letter appears exactly twice. Connect each letter
pair with an orthogonal path. A path may not revisit its own cells and may
not pass through any letter cell other than its two endpoints. Paths of
different letters MAY share cells. Objective: minimize cells covered by no
path.
"""

import random
import re
import string
try:                                        # ortools is only used by solve()/min_blanks_no_overlap.
    from ortools.sat.python import cp_model  # the webapp never solves at runtime (it grades against
except ImportError:                          # known optima), so the deployed app runs without ortools.
    cp_model = None


# ---------------------------------------------------------------- boards

def parse_board(text):
    """Parse a board from whitespace-separated rows of '_' and letters."""
    rows = [line.split() for line in text.strip().splitlines() if line.strip()]
    assert len({len(r) for r in rows}) == 1, "ragged board"
    return rows

def board_to_text(board):
    return "\n".join(" ".join(row) for row in board)

def letter_cells(board):
    """letter -> [(r, c), (r, c)] sorted by letter."""
    out = {}
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch != "_":
                out.setdefault(ch, []).append((r, c))
    for ch, cells in out.items():
        assert len(cells) == 2, f"letter {ch} appears {len(cells)} times"
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- generator

def generate(n_rows, n_cols, n_letters, rng, min_pair_dist=2):
    """Random board: place 2*n_letters letters on distinct cells, pairing so
    that no pair is orthogonally adjacent (avoids trivial 2-cell paths)."""
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for _ in range(200):
        picks = rng.sample(cells, 2 * n_letters)
        rng.shuffle(picks)
        pairs = [(picks[2 * i], picks[2 * i + 1]) for i in range(n_letters)]
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) >= min_pair_dist for a, b in pairs):
            board = [["_"] * n_cols for _ in range(n_rows)]
            for i, (a, b) in enumerate(pairs):
                board[a[0]][a[1]] = string.ascii_uppercase[i]
                board[b[0]][b[1]] = string.ascii_uppercase[i]
            return board
    return None


# ---------------------------------------------------------------- solver

def solve(board, time_limit_s=60):
    """Exact minimum number of uncovered cells, via CP-SAT.

    Returns dict: status ('OPTIMAL'|'FEASIBLE'|'INFEASIBLE'|'UNKNOWN'),
    min_blanks (int or None), paths (letter -> [(r,c), ...] or None).
    """
    n_rows, n_cols = len(board), len(board[0])
    letters = letter_cells(board)
    endpoint_of = {}  # cell -> letter
    for ch, cells in letters.items():
        for cell in cells:
            endpoint_of[cell] = ch

    model = cp_model.CpModel()
    visit = {}      # (letter, cell) -> BoolVar
    arc_lits = {}   # (letter, u, v) -> BoolVar  (real movement arcs)

    for ch, (s, t) in letters.items():
        # cells this path may occupy: anything that isn't another letter's endpoint
        allowed = [
            (r, c)
            for r in range(n_rows)
            for c in range(n_cols)
            if endpoint_of.get((r, c), ch) == ch
        ]
        idx = {cell: i for i, cell in enumerate(allowed)}
        arcs = []
        for cell in allowed:
            v = model.new_bool_var(f"v_{ch}_{cell}")
            visit[(ch, cell)] = v
            if cell in (s, t):
                model.add(v == 1)
            else:
                # self-loop selected <=> cell not on the path
                arcs.append((idx[cell], idx[cell], v.Not()))
        for (r, c) in allowed:
            for (r2, c2) in ((r + 1, c), (r, c + 1)):
                if (r2, c2) in idx:
                    for u, w in (((r, c), (r2, c2)), ((r2, c2), (r, c))):
                        lit = model.new_bool_var(f"a_{ch}_{u}_{w}")
                        arc_lits[(ch, u, w)] = lit
                        arcs.append((idx[u], idx[w], lit))
        # forced closing arc t -> s turns the s..t path into a circuit
        close = model.new_bool_var(f"close_{ch}")
        model.add(close == 1)
        arcs.append((idx[t], idx[s], close))
        model.add_circuit(arcs)

    # coverage objective over non-letter cells (letter cells are always covered)
    blanks = []
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in endpoint_of:
                continue
            vs = [visit[(ch, (r, c))] for ch in letters]
            cov = model.new_bool_var(f"cov_{r}_{c}")
            model.add_bool_or(vs).only_enforce_if(cov)
            for v in vs:
                model.add_implication(v, cov)
            blanks.append(cov.Not())
    model.minimize(sum(blanks))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    name = solver.status_name(status)
    if name not in ("OPTIMAL", "FEASIBLE"):
        return {"status": name, "min_blanks": None, "paths": None}

    paths = {}
    for ch, (s, t) in letters.items():
        nxt = {}
        for (ch2, u, w), lit in arc_lits.items():
            if ch2 == ch and solver.value(lit):
                nxt[u] = w
        path, cur = [s], s
        while cur != t:
            cur = nxt[cur]
            path.append(cur)
        paths[ch] = path
    return {"status": name, "min_blanks": int(solver.objective_value), "paths": paths}


# ---------------------------------------------------------------- checker

def check_answer(board, paths, optimal):
    """Validate model-supplied paths (1-indexed (row, col) tuples) and grade.

    Returns dict with valid, errors, achieved_blanks, correct.
    """
    n_rows, n_cols = len(board), len(board[0])
    letters = letter_cells(board)
    endpoint_of = {}
    for ch, cells in letters.items():
        for cell in cells:
            endpoint_of[cell] = ch

    errors = []
    covered = set()
    for ch, expected in letters.items():
        if ch not in paths:
            errors.append(f"{ch}: no path given")
            continue
        p = [(r - 1, c - 1) for r, c in paths[ch]]  # to 0-indexed
        if len(p) < 2:
            errors.append(f"{ch}: path too short")
            continue
        if not all(0 <= r < n_rows and 0 <= c < n_cols for r, c in p):
            errors.append(f"{ch}: cell out of bounds")
            continue
        if {p[0], p[-1]} != set(expected):
            errors.append(f"{ch}: endpoints {p[0]},{p[-1]} != letter cells")
            continue
        if len(set(p)) != len(p):
            errors.append(f"{ch}: path intersects itself")
            continue
        bad_adj = [
            (a, b) for a, b in zip(p, p[1:])
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
        ]
        if bad_adj:
            errors.append(f"{ch}: non-adjacent step {bad_adj[0]}")
            continue
        through = [cell for cell in p[1:-1] if cell in endpoint_of]
        if through:
            errors.append(f"{ch}: passes through letter at {through[0]}")
            continue
        covered.update(p)

    valid = not errors
    achieved = None
    if valid:
        achieved = n_rows * n_cols - len(covered)
    return {
        "valid": valid,
        "errors": errors,
        "achieved_blanks": achieved,
        "correct": valid and achieved == optimal,
    }


# ---------------------------------------------------------------- see_board render

def render_with_paths(board, paths):
    """Render the board with proposed paths, for a see_board tool.

    Endpoints show as the UPPERCASE letter; squares a path passes through show as that
    letter in lowercase; empty squares as '_'. On overlap, the letter listed FIRST in
    `paths` wins (matches the webapp's first-path-wins rule). `paths` is {letter:
    [(r,c) or [r,c], ...]} 1-indexed. Returns (text, None) or (None, error_string).
    """
    n_rows, n_cols = len(board), len(board[0])
    letters = letter_cells(board)
    endpoint_of = {cell: l for l, cs in letters.items() for cell in cs}
    if not isinstance(paths, dict) or not paths:
        return None, "provide paths as {letter: [[row,col], ...]}"
    norm = {}
    for ch, path in paths.items():
        if ch not in letters:
            return None, f"'{ch}' is not a letter in this puzzle"
        try:
            p = [(int(r) - 1, int(c) - 1) for r, c in path]
        except Exception:
            return None, f"{ch}: malformed path"
        if len(p) < 2:
            return None, f"{ch}: path too short"
        if not all(0 <= r < n_rows and 0 <= c < n_cols for r, c in p):
            return None, f"{ch}: a cell is out of bounds"
        if {p[0], p[-1]} != set(letters[ch]):
            return None, f"{ch}: endpoints are not the two {ch} squares"
        if len(set(p)) != len(p):
            return None, f"{ch}: path crosses itself"
        for a, b in zip(p, p[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                return None, f"{ch}: non-adjacent step ({a[0]+1},{a[1]+1})->({b[0]+1},{b[1]+1})"
        for cell in p[1:-1]:
            if cell in endpoint_of:
                return None, f"{ch}: passes through letter {endpoint_of[cell]} at ({cell[0]+1},{cell[1]+1})"
        norm[ch] = p
    disp = [row[:] for row in board]
    owned = {}
    for ch, p in norm.items():          # input order: first path wins on overlap
        for cell in p[1:-1]:
            owned.setdefault(cell, ch)
    for (r, c), ch in owned.items():
        disp[r][c] = ch.lower()
    empty = sum(row.count("_") for row in disp)
    return "\n".join(" ".join(row) for row in disp) + f"\n\nEmpty squares (no path): {empty}", None


# ---------------------------------------------------------------- answer parsing

def parse_final_answer(text):
    """Extract {letter: [(r,c), ...]} and claimed empty count from a response.

    Tolerant of separators (spaces, commas, arrows) between pairs.
    """
    tail = text
    m = None
    for m in re.finditer(r"FINAL ANSWER\s*:?", text, re.IGNORECASE):
        pass
    if m:
        tail = text[m.end():]
    paths = {}
    for line in tail.splitlines():
        lm = re.match(r"^\s*\**([A-Z])\**\s*[:=]\s*(.+)$", line.strip())
        if not lm:
            continue
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", lm.group(2))
        if len(pairs) >= 2:
            paths[lm.group(1)] = [(int(r), int(c)) for r, c in pairs]
    em = re.search(r"EMPTY(?:\s+SQUARES)?\s*[:=]?\s*\**\s*(\d+)", tail, re.IGNORECASE)
    claimed = int(em.group(1)) if em else None
    return paths, claimed


# ---------------------------------------------------------------- prompt

PROMPT_TEMPLATE = """\
Your goal is to connect the corresponding letters with paths, minimizing the number of squares without any path in them
Rules:
- a path moves between orthogonally adjacent squares (up/down/left/right) and may not revisit any of its own squares (no self-intersection)
- a path may not pass through any letter/endpoint square other than its own two matching endpoints
- **Different letters' paths MAY OVERLAP — this is explicitly allowed and is NOT a conflict.** One and the same empty square can be used by MULTIPLE letters' paths at once — two, three, or more different letters may all route through it simultaneously. For example, empty square (2,2) may be part of A's path AND B's path AND C's path at the same time, and that is perfectly legal. (A square is "empty" only if NO path uses it; a square shared by one or several paths counts as covered. Overlapping costs you nothing.)

No code or external tools allowed — solve by pure reasoning.

Think as much as you'd like, then return your final answer as a list of ordered pairs describing the path for A, B, C, ..., respectively, along with the number of squares without a path in them.

Use (row, column) coordinates: row 1 is the top row, column 1 is the leftmost column. Each path must start at one copy of its letter and end at the other copy, and must include both endpoint squares.

End your response with exactly this format:

FINAL ANSWER:
A: (r,c) (r,c) ...
B: (r,c) (r,c) ...
...
EMPTY: <number of squares without any path>

Board:
{board}
"""

def make_prompt(board):
    return PROMPT_TEMPLATE.format(board=board_to_text(board))
