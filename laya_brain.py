"""Laya brain for the Snake game - the part that decides where the snake goes.

``clever_snake.py`` describes the board, asks this module "which arrow should I
press next?" and then presses the answer (Up / Down / Left / Right).  The
question is an ordinary Laya ``choice`` question, so the answer arrives as
``result["answers"]["direction"]["choice"]``:

    import laya_brain

    brain = laya_brain.LayaBrain()
    brain.start()                        # loads the model on a worker thread
    brain.submit(key, board)             # key = fingerprint of that board
    direction = brain.take_answer(key)   # "Up" / "Down" / "Left" / "Right"
    safe = brain.resolve(board, direction)   # never into death, always at the food

Two backends, pick one with ``BACKEND`` below:

    "local"  (default) imports the ``laya`` package and runs
             "convaiinnovations/laya-multilingual" inside this process.
    "http"   POSTs ``{"text": board_text, "questions": QUESTIONS}`` to a
             FastAPI service at ``API_URL``.  The server side of that is the
             sample call, made generic:

                 from fastapi import FastAPI
                 from pydantic import BaseModel
                 import laya

                 agent = laya.load("convaiinnovations/laya-multilingual")
                 app = FastAPI()

                 class Request(BaseModel):
                     text: str
                     questions: dict = {}

                 @app.post("/predict")
                 def predict(request: Request):
                     return agent.predict({"body": request.text},
                                          request.questions)

Everything runs on a background thread, so the Tkinter window never freezes
and the game keeps playing while the model is still loading.  Laya steers, but
it is not allowed to kill the snake or to lose sight of the goal: a move into a
wall, the body or a U-turn back into the neck is quietly replaced by a
built-in policy, and so is a legal move that walks away from the food while a
route to it is still open (see ``SAFETY_FILTER`` and ``GOAL_FILTER``).  That
same policy drives the snake whenever the model cannot be used at all, and the
reason is reported through :attr:`LayaBrain.status`.

A model this size needs about a second per answer on a CPU, which is far
slower than the game itself.  So the game does not wait for it: it keeps its
own pace and treats each answer as a heading - :meth:`take_answer` when the
answer happens to be for the exact board on screen, :meth:`take_latest` for
the newest one otherwise, and :meth:`resolve` to make it safe _and_ useful for
the board the snake is on right now.  A caller that would rather have every
single move be the model's own can simply stop until ``take_answer`` returns.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------
# Configuration - tweak these to point the brain somewhere else
# ---------------------------------------------------------------------------

BACKEND = "local"                                   # "local" or "http"
LOCAL_MODEL = "convaiinnovations/laya-multilingual"  # any Laya checkpoint
API_URL = "http://127.0.0.1:8000/predict"           # used by BACKEND = "http"
HTTP_TIMEOUT = 30.0                                 # seconds, network only

IDLE_SLEEP = 0.005      # worker nap while there is no board to look at
KEEP_ANSWERS = 4        # remembered answers; older states can never be asked

# Laya is a small "System 1" reasoner: most of the time it picks a sane arrow,
# but now and then it asks for a wall, its own body or a U-turn back into its
# neck.  The game refuses U-turns and dies on the rest, so while this is on
# those answers are replaced by the built-in safety policy.  Switch it off to
# watch the raw model play (and lose).
SAFETY_FILTER = True

# The model is also easily distracted: measured on real games, it often picks
# an arrow that is perfectly legal and yet points nowhere near the food.  With
# this on, such an arrow is swapped for the step the built-in goal policy would
# take (the one that shortens the route to the food), so the snake still hunts
# while Laya keeps deciding every other move.  Switch it off to follow the
# model's aim as-is - safety still applies.
GOAL_FILTER = True

DIRECTIONS = ("Up", "Down", "Left", "Right")

DIRECTION_DELTAS = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}

# The question the game asks about every board.  Criteria are the four arrow
# keys, so Laya's answer is directly a key to press.
QUESTIONS = {
    "direction": {
        "type": "choice",
        "instructions": (
            "You are the brain of a snake on a grid. The goal is to eat the "
            "food. Decide where the snake moves next: pick the arrow that "
            "takes it one cell closer to the food along the shortest route, "
            "without dying. Never move into a wall or into the snake's own "
            "body, and never turn back the way you came. If no route to the "
            "food is open, choose the move that keeps the snake alive the "
            "longest."
        ),
        "criteria": {
            "Up": "Move one cell up (y decreases by 1).",
            "Down": "Move one cell down (y increases by 1).",
            "Left": "Move one cell left (x decreases by 1).",
            "Right": "Move one cell right (x increases by 1).",
        },
    }
}


# ---------------------------------------------------------------------------
# Board -> text (this is what Laya actually reads)
# ---------------------------------------------------------------------------

def _direction_name(delta):
    """Turn a (dx, dy) step into "Up" / "Down" / "Left" / "Right"."""
    for name, known in DIRECTION_DELTAS.items():
        if tuple(delta) == known:
            return name
    return "Right"


def _relative_position(head, cell):
    """Describe ``cell`` from the head, e.g. "above and right of the head"."""
    dx = cell[0] - head[0]
    dy = cell[1] - head[1]
    parts = []
    if dy < 0:
        parts.append("above")
    elif dy > 0:
        parts.append("below")
    if dx < 0:
        parts.append("left")
    elif dx > 0:
        parts.append("right")
    if not parts:
        return "on the head"
    return " and ".join(parts) + " of the head"


def _blocked_cells(board):
    """Cells the snake body occupies - without the tail tip, which moves away."""
    return set(map(tuple, board["snake"][:-1]))


def safe_directions(board):
    """Directions that do not kill the snake this step, in arrow-key order.

    Walls, the body, and a 180-degree turn (the game refuses those) are out.
    """
    cols, rows = board["cols"], board["rows"]
    head_col, head_row = board["snake"][0]
    current = tuple(board["direction"])
    reverse = (-current[0], -current[1])
    blocked = _blocked_cells(board)

    safe = []
    for name in DIRECTIONS:
        delta = DIRECTION_DELTAS[name]
        if delta == reverse:
            continue
        cell = (head_col + delta[0], head_row + delta[1])
        if not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
            continue
        if cell in blocked:
            continue
        safe.append(name)
    return safe


def _reachable_area(board, start):
    """How many free cells the snake can still reach from ``start`` (BFS)."""
    cols, rows = board["cols"], board["rows"]
    blocked = _blocked_cells(board)
    if start in blocked:
        return 0

    seen = {start}
    queue = [start]
    while queue:
        cell = queue.pop()
        for delta in DIRECTION_DELTAS.values():
            neighbour = (cell[0] + delta[0], cell[1] + delta[1])
            if neighbour in seen or neighbour in blocked:
                continue
            if not (0 <= neighbour[0] < cols and 0 <= neighbour[1] < rows):
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return len(seen)


def describe_board(board):
    """Render the board snapshot as the short text prompt Laya reads."""
    cols, rows = board["cols"], board["rows"]
    snake = board["snake"]
    head_col, head_row = snake[0]
    lines = [
        "Snake board: %d columns (x = 0..%d) and %d rows (y = 0..%d); "
        "x grows to the right, y grows downwards."
        % (cols, cols - 1, rows, rows - 1),
        "Snake head: x = %d, y = %d. Length: %d. Currently moving %s."
        % (head_col, head_row, len(snake), _direction_name(board["direction"])),
    ]

    food = board["food"]
    if food is None:
        lines.append("Goal: eat the food - there is none left on the board.")
    else:
        distance = abs(food[0] - head_col) + abs(food[1] - head_row)
        steps = _food_distances(board).get((head_col, head_row))
        route = (
            "the shortest route there is %d steps long" % steps
            if steps is not None
            else "no route there is open right now"
        )
        lines.append(
            "Goal - the food: x = %d, y = %d (%s, %d cells away); %s."
            % (food[0], food[1], _relative_position((head_col, head_row), food),
               distance, route)
        )

    body_near = [
        cell for cell in map(tuple, snake[1:])
        if abs(cell[0] - head_col) <= 1 and abs(cell[1] - head_row) <= 1
    ]
    if body_near:
        lines.append(
            "Body cells touching the head (deadly): %s."
            % ", ".join("x = %d, y = %d" % cell for cell in body_near)
        )
    tail_col, tail_row = snake[-1]
    lines.append("Tail tip: x = %d, y = %d." % (tail_col, tail_row))

    safe = safe_directions(board)
    lines.append(
        "Safe moves right now: %s."
        % (", ".join(safe) if safe else "none, the snake is trapped")
    )
    back = (-board["direction"][0], -board["direction"][1])
    lines.append(
        "Never choose %s - that is straight back into the snake's own neck."
        % _direction_name(back)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in goal policy (fallback driver, and the yardstick for Laya's answers)
# ---------------------------------------------------------------------------

def _food_distances(board):
    """Steps from every free cell to the food, by BFS.

    The food is the goal, so one breadth-first search out from it tells every
    cell how far it still has to go.  Walls and the snake's body block the
    way; the tail tip does not, because it moves on, and neither does the head
    - the snake is about to leave it.
    """
    food = board["food"]
    if food is None:
        return {}
    cols, rows = board["cols"], board["rows"]
    blocked = _blocked_cells(board) - {tuple(board["snake"][0])}
    start = tuple(food)
    if start in blocked:
        return {}

    distances = {start: 0}
    queue = [start]
    index = 0
    while index < len(queue):
        cell = queue[index]
        index += 1
        for delta in DIRECTION_DELTAS.values():
            neighbour = (cell[0] + delta[0], cell[1] + delta[1])
            if neighbour in distances or neighbour in blocked:
                continue
            if not (0 <= neighbour[0] < cols and 0 <= neighbour[1] < rows):
                continue
            distances[neighbour] = distances[cell] + 1
            queue.append(neighbour)
    return distances


def _move_quality(board, name, distances, length):
    """How good one arrow is for reaching the goal.

    First, does the move leave the snake room to breathe (at least its own
    length in free cells)?  Then, can the food still be reached from the cell
    it lands on, and if so how soon?  Two arrows of equal quality are equally
    good ways of chasing the food, whichever one is picked.
    """
    head_col, head_row = board["snake"][0]
    delta = DIRECTION_DELTAS[name]
    cell = (head_col + delta[0], head_row + delta[1])
    room = _reachable_area(board, cell)
    steps = distances.get(cell)
    return (room >= length, steps is not None, -(steps or 0))


def _goal_choice(board):
    """The arrow the goal policy would press: chase the food, never get boxed in.

    Of the safe moves, prefer one that leaves the snake room to breathe, then
    the one that shortens the route to the food most, then simply carrying on
    straight.  The route is walked with BFS, so the snake goes around its own
    body instead of pushing against it.
    """
    options = safe_directions(board)
    current = _direction_name(board["direction"])
    if not options:
        return current                      # nothing can save the snake

    distances = _food_distances(board)
    length = len(board["snake"])
    return max(
        options,
        key=lambda name: (_move_quality(board, name, distances, length),
                          name == current),
    )


def _same_goal(board, first, second):
    """True when both arrows chase the food equally well (see _move_quality)."""
    if first == second:
        return True
    distances = _food_distances(board)
    length = len(board["snake"])
    return (_move_quality(board, first, distances, length)
            == _move_quality(board, second, distances, length))


# ---------------------------------------------------------------------------
# Backends - the two ways of asking Laya
# ---------------------------------------------------------------------------

def _answer_direction(result):
    """Pull "Up"/"Down"/"Left"/"Right" out of a Laya response."""
    answer = ((result or {}).get("answers") or {}).get("direction") or {}

    choice = answer.get("choice")
    if isinstance(choice, str):
        for name in DIRECTIONS:
            if choice.strip().lower() == name.lower():
                return name

    # If the model returned something unexpected, trust its probabilities.
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        best = max(probabilities.items(), key=lambda item: item[1])
        for name in DIRECTIONS:
            if str(best[0]).strip().lower() == name.lower():
                return name
    return None


class LocalLayaBackend:
    """Runs the Laya checkpoint inside this process (``import laya``)."""

    name = "local"

    def __init__(self, model=LOCAL_MODEL):
        self.model = model
        self._agent = None

    def prepare(self):
        """Load the model - takes a while, so it happens on the worker."""
        import laya

        self._agent = laya.load(self.model)

    def ask(self, board):
        result = self._agent.predict({"body": describe_board(board)}, QUESTIONS)
        return _answer_direction(result)


class HttpLayaBackend:
    """Asks a FastAPI service that wraps the same model (see module docstring)."""

    name = "http"

    def __init__(self, url=API_URL, timeout=HTTP_TIMEOUT):
        self.url = url
        self.timeout = timeout

    def prepare(self):
        """Nothing to load here - the service owns the model."""

    def ask(self, board):
        payload = json.dumps({
            "text": describe_board(board),
            "questions": QUESTIONS,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _answer_direction(result)


def _make_backend():
    if str(BACKEND).lower() == "http":
        return HttpLayaBackend()
    return LocalLayaBackend()


# ---------------------------------------------------------------------------
# The brain
# ---------------------------------------------------------------------------

class LayaBrain:
    """Answers "which arrow next?" without ever blocking the caller.

    ``submit`` hands over a board, ``take_answer`` picks up the arrow once the
    worker thread has one.  Only the newest board is kept waiting, and answers
    are filed under the board they were computed for, so a slow model can
    never move the snake using an outdated decision.
    """

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else _make_backend()
        self._lock = threading.Lock()
        self._pending = None        # (key, board) waiting for the worker
        self._inflight = None       # key the worker is busy with
        self._answers = {}          # key -> direction, newest last
        self._status = "idle"       # idle | loading | ready | fallback
        self._detail = ""           # why it fell back, for the console
        self._needs_prepare = True
        self._worker = None
        self._stop = threading.Event()
        self.filtered = 0           # answers replaced by the safety policy
        self.redirected = 0         # safe answers replaced by the goal policy

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        """Start the worker thread (and retry the model after a failure)."""
        with self._lock:
            self._needs_prepare = True
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._status = "loading"
            self._worker = threading.Thread(
                target=self._run, name="laya-brain", daemon=True,
            )
            self._worker.start()

    def stop(self):
        """Ask the worker to finish; safe to call at any time."""
        self._stop.set()
        with self._lock:
            self._pending = None

    # -- API used by the game ------------------------------------------------
    def submit(self, key, board):
        """Queue ``board`` under ``key``; replaces any older waiting board."""
        with self._lock:
            if key == self._inflight or key in self._answers:
                return
            self._pending = (key, board)

    def take_answer(self, key):
        """Return the direction computed for ``key``, or None if not ready."""
        with self._lock:
            return self._answers.pop(key, None)

    def take_latest(self):
        """Return the newest answer, whatever board it was computed for.

        A driver that keeps its own pace reads headings this way: the answer
        describes how Laya wanted the snake to turn a moment ago, and
        :meth:`resolve` checks it against the board on screen now.
        """
        with self._lock:
            if not self._answers:
                return None
            newest = next(reversed(self._answers))
            return self._answers.pop(newest)

    def resolve(self, board, direction):
        """Make ``direction`` suit ``board``: legal, and aimed at the goal.

        This is the net the game relies on while Laya drives.  The built-in
        policy answers instead of a heading that would hit a wall, the body or
        the snake's own neck, when there is no answer yet - and, with
        ``GOAL_FILTER`` on, instead of a legal heading that walks away from the
        food while a route to it is still open.
        """
        plan = _goal_choice(board)
        if direction is None:
            return plan
        if direction not in safe_directions(board):
            if not SAFETY_FILTER:
                return direction            # raw model: it may crash
            self.filtered += 1
            return plan
        if GOAL_FILTER and not _same_goal(board, direction, plan):
            self.redirected += 1
            return plan
        return direction

    @property
    def status(self):
        """One of "idle", "loading", "ready" or "fallback"."""
        with self._lock:
            return self._status

    @property
    def detail(self):
        """Why Laya fell back, if it did (handy while debugging)."""
        with self._lock:
            return self._detail

    @property
    def backend_name(self):
        return getattr(self._backend, "name", "?")

    # -- worker thread ------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            item = self._take_pending()
            if item is None:
                time.sleep(IDLE_SLEEP)
                continue

            key, board = item
            direction = self._decide(board)

            with self._lock:
                self._inflight = None
                if direction is not None:
                    self._answers[key] = direction
                    while len(self._answers) > KEEP_ANSWERS:
                        self._answers.pop(next(iter(self._answers)))

    def _take_pending(self):
        with self._lock:
            if self._pending is None:
                return None
            item = self._pending
            self._pending = None
            self._inflight = item[0]
            return item

    def _decide(self, board):
        """One answer: Laya when it is available, the goal policy if not."""
        with self._lock:
            prepare = self._needs_prepare
            self._needs_prepare = False
        if prepare:
            self._prepare_backend()

        if self.status == "ready":
            try:
                direction = self._backend.ask(board)
                if direction is not None:
                    return self._keep_safe(board, direction)
                self._set_status("fallback", "Laya answered with an unknown move")
            except Exception as exc:
                self._set_status("fallback", "%s: %s" % (type(exc).__name__, exc))

        return _goal_choice(board)

    def _keep_safe(self, board, direction):
        """Let Laya steer, but never let it steer the snake into death.

        Only the safety net runs here, on the board the answer was computed
        for; whether the arrow still chases the goal is judged by
        :meth:`resolve`, against the board the snake is on when it moves.
        """
        if not SAFETY_FILTER or direction in safe_directions(board):
            return direction
        self.filtered += 1
        return _goal_choice(board)

    def _prepare_backend(self):
        self._set_status("loading")
        try:
            self._backend.prepare()
        except Exception as exc:
            self._set_status("fallback", "%s: %s" % (type(exc).__name__, exc))
        else:
            self._set_status("ready")

    def _set_status(self, status, detail=""):
        with self._lock:
            self._status = status
            if detail:
                self._detail = detail
