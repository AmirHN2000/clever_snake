# Clever Snake

A classic arcade Snake built with Python 3 and Tkinter, with an optional AI
pilot: flip one switch and the **Laya** model presses the arrow keys for you.

```
python clever_snake.py
```

## Requirements

- Python 3.8+ with Tkinter (bundled with the official installers).
- The game itself is **standard library only** — no `pip install` needed.
- Optional, for the Laya switch: the `laya` package and the
  `convaiinnovations/laya-multilingual` checkpoint
  (see <https://huggingface.co/convaiinnovations/laya>).  It downloads the
  model on first use and runs it on the CPU.

## Files

| File | Purpose |
| --- | --- |
| `clever_snake.py` | The whole game: rules, Tkinter GUI, keyboard input, difficulty. |
| `laya_brain.py` | The AI pilot. Separate module, loaded only if present. |

If `laya_brain.py` is missing or broken, the game still runs and the Laya
switch is simply disabled.

## Controls

| Key | Action |
| --- | --- |
| Arrow keys / WASD | Move the snake (no immediate reversal). |
| Space | Pause / resume. |
| Enter | Start the game / play again after a game over. |
| R | Restart at any time. |

There are also buttons for **Start Game**, **Pause**/**Resume**,
**Restart**, the difficulty (**Easy** / **Medium** / **Hard**) and the
**Laya** switch.

## Features

- 28 x 20 cell board on a dark 600 x 600 window, rounded grid-aligned snake
  and a distinct food diamond.
- Eating grows the snake, raises the score and speeds the game up; the delay
  per step never falls below the difficulty's minimum.
- Score and session high score shown above the board.
- Start screen with the `SNAKE` title and instructions; 3-2-1-GO countdown;
  game-over screen with final score and a working Restart.
- Pause is clearly indicated and freezes everything.
- Optional beeps through `winsound` (stdlib, Windows only; other systems
  simply stay silent).
- Movement is driven by Tkinter's `after()` — never two timers at once,
  never `while True`.

## Difficulties

| Preset | Step delay | Speeds up every | Minimum delay |
| --- | --- | --- | --- |
| Easy | 190 ms | 4 points | 115 ms |
| Medium (default) | 145 ms | 3 points | 80 ms |
| Hard | 105 ms | 2 points | 55 ms |

## Laya mode (automatic play)

Flip the **Laya** switch in the footer. The status label next to it tells you
what is happening:

| Label | Meaning |
| --- | --- |
| `Laya: Off` | The switch is off; you drive. |
| `Laya: N/A` | `laya_brain.py` (or its dependencies) is not available. |
| `Laya: Loading` | The model is loading on a background thread (~12 s); the built-in policy plays meanwhile. |
| `Laya: On` | The model is answering and steering. |
| `Laya: Fallback` | The model failed; the built-in policy plays. |

The board is turned into a short text description, the question *"which arrow
should the snake press?"* is an ordinary Laya `choice` question, and the
answer is one of `Up` / `Down` / `Left` / `Right`. A small badge above the
board shows what happened on the last step:

| Badge | Meaning |
| --- | --- |
| `Laya ↑ Up` (accent colour) | Laya's arrow was safe and on-goal, and it was pressed. |
| `Laya ↑ Up → Right` (gold) | Laya said `Up`, but a filter replaced it with `Right`. |

### How one move is decided

1. The game keeps its own difficulty pace — it never waits for the model.
   A model answer takes ~1 s on a CPU, far slower than a Hard step (105 ms).
2. Each answer is treated as a **heading**: the game uses the answer for the
   exact board on screen when there is one, otherwise the newest answer.
3. Before the heading becomes a key press, `LayaBrain.resolve()` checks it
   against the board the snake is on *right now*:
   - **safety** (`SAFETY_FILTER`) — a move into a wall, the body or a U-turn
     is replaced by the built-in policy (counted in `brain.filtered`);
   - **goal** (`GOAL_FILTER`) — a legal move that walks away from the food
     while a route to it is still open is replaced too (counted in
     `brain.redirected`).
4. The built-in policy is a BFS planner: it searches outward from the food,
   prefers moves that keep at least the snake's own length in reachable free
   space, then the move that shortens the route to the food.

Measured on the real checkpoint, this makes the difference between a snake
that wanders and one that hunts: 90 s on Hard scored **67** points with the
filters on, versus **1** with a raw model driving. The checkpoint answers
"Left" on 9 of 10 boards regardless of how the board is described, so the
filters are what keep the snake alive and fed; set
`SAFETY_FILTER = False` / `GOAL_FILTER = False` in `laya_brain.py` to watch
the raw behaviour instead.

Switching Laya off at any time hands the keys straight back to you.

## Configuration

Everything worth tweaking sits in a marked block at the top of each file.

`clever_snake.py`:

- `CELL_SIZE`, `GRID_COLS`, `GRID_ROWS` — board geometry.
- `INITIAL_LENGTH` — starting snake length.
- `DIFFICULTIES` — delay, speed-up rate and minimum delay per preset.
- `DEFAULT_DIFFICULTY` — which preset is selected at launch.

`laya_brain.py`:

- `BACKEND` — `"local"` runs the model in-process; `"http"` calls a server.
- `LOCAL_MODEL` — any Laya checkpoint id.
- `API_URL`, `HTTP_TIMEOUT` — used by the `"http"` backend.
- `SAFETY_FILTER`, `GOAL_FILTER` — the two rules described above.
- `KEEP_ANSWERS` — how many recent answers are remembered.
- `IDLE_SLEEP` — worker nap when there is no board to look at.

### Using the HTTP backend

Set `BACKEND = "http"` in `laya_brain.py` and run this service next to the
game (the sample FastAPI call, made generic):

```python
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
    return agent.predict({"body": request.text}, request.questions)
```

The game POSTs `{"text": board_text, "questions": QUESTIONS}` and expects the
Laya result JSON back.

## Troubleshooting

- **The first Laya game starts slowly.** The checkpoint (~12 s on a CPU)
  loads on a worker thread; the snake plays on the built-in policy and the
  label reads `Laya: Loading`. Nothing freezes.
- **The console shows `?` instead of `↑` on Windows.** The console codepage
  is cp1252; run with `set PYTHONIOENCODING=utf-8` (or just ignore it — the
  window itself is fine).
- **No sound.** `winsound` exists on Windows only; the game is silent
  elsewhere by design.
- **`Laya: N/A`.** `laya_brain.py` could not be imported — check that it sits
  next to `clever_snake.py` and that the `laya` package is installed.
