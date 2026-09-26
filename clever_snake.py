"""Clever Snake - a classic arcade Snake built with Python 3 and Tkinter.

The game itself only uses the standard library, so it runs anywhere Python
(with Tkinter) is installed:

    python clever_snake.py

Next to it lives ``laya_brain.py``, an optional module that lets the Laya
model (https://huggingface.co/convaiinnovations/laya) play the game for you:
flip the "Laya" switch in the footer and the snake asks Laya which arrow to
press before every step.  Without that file the switch is simply disabled.

Controls
    Arrow keys / WASD ... move the snake
    Space .............. pause / resume
    Enter .............. start / play again
    R .................. restart
"""

from __future__ import annotations

import random
import tkinter as tk
from collections import deque
from tkinter import font as tkfont

try:
    # winsound ships with the standard library, but only exists on Windows.
    import winsound
except ImportError:  # pragma: no cover - Linux / macOS simply stay silent
    winsound = None

try:
    # Optional: the Laya brain that can play the game on its own.
    from laya_brain import LayaBrain
except Exception:  # placeholder: the module (or its dependencies) is missing
    LayaBrain = None


# ---------------------------------------------------------------------------
# Configuration - tweak these values to change how the game feels
# ---------------------------------------------------------------------------

CELL_SIZE = 20          # Size of a single grid cell, in pixels.
GRID_COLS = 28          # Playfield width, in cells.
GRID_ROWS = 20          # Playfield height, in cells.
INITIAL_LENGTH = 4      # Segments the snake starts with.

BOARD_WIDTH = GRID_COLS * CELL_SIZE
BOARD_HEIGHT = GRID_ROWS * CELL_SIZE
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Difficulty presets.  "delay" is the number of milliseconds between two snake
# steps; every "speedup_every" points the delay drops by "speedup_step" ms and
# never falls below "min_delay".  That is the "gets harder as you grow" rule.
DIFFICULTIES = {
    "Easy": {"delay": 190, "min_delay": 115, "speedup_every": 4, "speedup_step": 6},
    "Medium": {"delay": 145, "min_delay": 80, "speedup_every": 3, "speedup_step": 6},
    "Hard": {"delay": 105, "min_delay": 55, "speedup_every": 2, "speedup_step": 6},
}
DEFAULT_DIFFICULTY = "Medium"

COUNTDOWN_FROM = 3          # The "3, 2, 1, GO!" countdown before every round.
COUNTDOWN_TICK_MS = 650     # Milliseconds each countdown number stays visible.

DIRECTIONS = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}

# Both the arrow keys and WASD are accepted (lower-cased keysym values).
KEY_TO_DIRECTION = {
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "w": "Up", "s": "Down", "a": "Left", "d": "Right",
}

# --- palette: a clean, modern dark theme ----------------------------------
COLOR_BG = "#0f1420"
COLOR_BOARD = "#0a1018"
COLOR_GRID = "#141d2e"
COLOR_BORDER = "#1e2a44"
COLOR_PANEL = "#111a2b"
COLOR_BUTTON = "#1b2740"
COLOR_BUTTON_ACTIVE = "#26344f"
COLOR_TEXT = "#e6edf7"
COLOR_MUTED = "#8b96ac"
COLOR_ACCENT = "#38bdf8"
COLOR_SNAKE_HEAD = "#7bf1a8"
COLOR_SNAKE_TAIL = "#15803d"
COLOR_EYE = "#08111d"
COLOR_FOOD = "#f43f5e"
COLOR_FOOD_SHINE = "#ffc9d4"
COLOR_LEAF = "#4ade80"
COLOR_DANGER = "#fb7185"
COLOR_GOLD = "#fbbf24"

# --- Laya auto-pilot -------------------------------------------------------
# When the Laya switch is on, laya_brain decides the direction of every step
# instead of the keyboard.  The snake does not wait for it: steps keep being
# played at the usual difficulty pace, Laya's newest answer is followed as a
# heading, and the built-in safety policy drives the steps where the model is
# still thinking or asks for something that would kill the snake.  A CPU model
# needs around a second per answer, so waiting for it would freeze the game on
# Hard - this way the difficulty still sets the speed.
LAYA_GLYPHS = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→"}

# --- game states -----------------------------------------------------------
STATE_START = "start"
STATE_COUNTDOWN = "countdown"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_GAME_OVER = "game_over"
STATE_WON = "won"


# ---------------------------------------------------------------------------
# Game logic (no Tkinter in here on purpose, so the rules stay easy to read)
# ---------------------------------------------------------------------------

class SnakeGame:
    """The rules of Snake: the body, the food, the score and collisions.

    Coordinates are grid cells given as ``(column, row)`` tuples.  The head of
    the snake is always ``snake[0]``.
    """

    def __init__(self, cols=GRID_COLS, rows=GRID_ROWS, initial_length=INITIAL_LENGTH):
        self.cols = cols
        self.rows = rows
        self.initial_length = initial_length
        self.reset()

    # -- setup --------------------------------------------------------------
    def reset(self):
        """Put the snake back in its starting spot and zero the score."""
        length = max(1, min(self.initial_length, self.cols * self.rows))
        head_col = self.cols // 2
        head_row = self.rows // 2

        body = []
        for index in range(length):
            cell = (max(0, min(head_col - index, self.cols - 1)), head_row)
            if cell not in body:  # keeps tiny test boards sane
                body.append(cell)

        self.snake = deque(body)
        self.direction = (1, 0)     # heading right
        self.turn_queue = deque()   # at most two buffered turns
        self.score = 0
        self.alive = True
        self.food = self.spawn_food()

    def spawn_food(self):
        """Return a random cell that is not covered by the snake."""
        occupied = set(self.snake)
        free_cells = [
            (col, row)
            for row in range(self.rows)
            for col in range(self.cols)
            if (col, row) not in occupied
        ]
        return random.choice(free_cells) if free_cells else None

    # -- helpers ------------------------------------------------------------
    @property
    def head(self):
        return self.snake[0]

    @property
    def length(self):
        return len(self.snake)

    def is_inside(self, cell):
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows

    # -- actions ------------------------------------------------------------
    def turn(self, direction_name):
        """Queue a direction change for the next steps.

        Reversing straight into the neck, repeating the current direction and
        a full turn queue are all ignored.  Returns True if it was accepted.
        """
        new_direction = DIRECTIONS[direction_name]
        last = self.turn_queue[-1] if self.turn_queue else self.direction

        if new_direction == last:
            return False
        if new_direction[0] == -last[0] and new_direction[1] == -last[1]:
            return False
        if len(self.turn_queue) >= 2:
            return False

        self.turn_queue.append(new_direction)
        return True

    def step(self):
        """Advance the snake by one cell.

        Returns one of: "ok", "eat", "wall", "self", "win".
        """
        if self.turn_queue:
            self.direction = self.turn_queue.popleft()

        head_col, head_row = self.snake[0]
        delta_col, delta_row = self.direction
        new_head = (head_col + delta_col, head_row + delta_row)

        if not self.is_inside(new_head):
            self.alive = False
            return "wall"

        eating = new_head == self.food
        # The tail cell frees up in this same step, unless the snake grows.
        body = self.snake if eating else list(self.snake)[:-1]
        if new_head in body:
            self.alive = False
            return "self"

        self.snake.appendleft(new_head)
        if not eating:
            self.snake.pop()
            return "ok"

        self.score += 1
        if len(self.snake) >= self.cols * self.rows:
            self.food = None    # board completely filled: nothing left to eat
            self.alive = False
            return "win"

        self.food = self.spawn_food()
        return "eat"


# ---------------------------------------------------------------------------
# Sound (optional, Windows only, never a hard requirement)
# ---------------------------------------------------------------------------

class SoundPlayer:
    """Minimal wrapper around the standard-library ``winsound`` module.

    On non-Windows systems, or if a sound ever fails, every call becomes a
    no-op so audio can never break the game.
    """

    ALIASES = {
        "start": "SystemAsterisk",
        "eat": "SystemQuestion",
        "game_over": "SystemHand",
        "click": "SystemDefault",
    }

    def __init__(self):
        self.enabled = winsound is not None

    def toggle(self):
        """Flip mute on/off and return the new enabled state."""
        if winsound is None:
            return False
        self.enabled = not self.enabled
        return self.enabled

    def play(self, name):
        if not self.enabled or winsound is None:
            return
        alias = self.ALIASES.get(name)
        if alias is None:
            return
        try:
            winsound.PlaySound(
                alias,
                winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        except Exception:
            self.enabled = False   # audio is a nice-to-have, never fatal


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SnakeApp(tk.Tk):
    """The window: header, board canvas, footer buttons and the game loop."""

    def __init__(self):
        super().__init__()

        self.title("Clever Snake")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.font_family = self._pick_font_family()
        self.sound = SoundPlayer()
        self.game = SnakeGame()

        self.brain = LayaBrain() if LayaBrain is not None else None
        self.laya_on = False
        self._laya_direction = None    # the arrow the brain picked last
        self._laya_executed = None     # the arrow the snake actually took
        self._laya_button_text = None

        self.difficulty = DEFAULT_DIFFICULTY
        self.high_score = 0
        self.new_record = False

        self.state = STATE_START
        self._loop_id = None          # id of the scheduled game step
        self._countdown_id = None     # id of the scheduled countdown tick
        self._overlay_widgets = []    # real widgets placed on top of the canvas
        self._difficulty_buttons = {}

        self._build_ui()
        self._draw_board()
        self.bind("<Key>", self._on_key)
        self._center_window()
        self._update_header()
        self._update_controls()
        self._show_start_screen()
        self.canvas.focus_set()

    # -- construction --------------------------------------------------------
    def _pick_font_family(self):
        """Use a nice UI font when the platform has one, otherwise the default."""
        available = set(tkfont.families(self))
        for candidate in ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Arial"):
            if candidate in available:
                return candidate
        return "TkDefaultFont"

    def _build_stat(self, parent, caption, value_var, color):
        """One "label + big number" block for the header."""
        box = tk.Frame(parent, bg=COLOR_BG)
        tk.Label(box, text=caption, bg=COLOR_BG, fg=COLOR_MUTED,
                 font=(self.font_family, 8, "bold")).pack(anchor="w")
        tk.Label(box, textvariable=value_var, bg=COLOR_BG, fg=color,
                 font=(self.font_family, 16, "bold")).pack(anchor="w")
        return box

    def _make_button(self, parent, text, command, width=10, bg=None, fg=None):
        """Create a flat, styled button that never steals keyboard focus."""
        def invoke():
            command()
            self.canvas.focus_set()   # keep Space/arrows working after a click

        return tk.Button(
            parent,
            text=text,
            command=invoke,
            font=(self.font_family, 10, "bold"),
            bg=bg or COLOR_BUTTON,
            fg=fg or COLOR_TEXT,
            activebackground=COLOR_ACCENT if bg else COLOR_BUTTON_ACTIVE,
            activeforeground="#04121f" if bg else COLOR_TEXT,
            disabledforeground=COLOR_MUTED,
            relief="flat",
            bd=0,
            width=width,
            padx=8,
            pady=6,
            cursor="hand2",
            takefocus=0,   # so the Space key can never press a focused button
        )

    def _build_ui(self):
        """Header on top, board in the middle, controls at the bottom."""
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(side="top", fill="x", padx=20, pady=(14, 6))

        self.score_var = tk.StringVar(value="0")
        self.high_var = tk.StringVar(value="0")
        self.mode_var = tk.StringVar(value=self.difficulty)
        self.speed_var = tk.StringVar(value="1")

        self._build_stat(header, "SCORE", self.score_var, COLOR_SNAKE_HEAD).pack(side="left")
        self._build_stat(header, "BEST", self.high_var, COLOR_GOLD).pack(side="left", padx=(28, 0))
        self._build_stat(header, "SPEED", self.speed_var, COLOR_TEXT).pack(side="right")
        self._build_stat(header, "MODE", self.mode_var, COLOR_ACCENT).pack(side="right", padx=(0, 28))

        # The footer is packed before the board so the board only gets what is
        # left between them, which keeps the playfield centred in the window.
        footer = tk.Frame(self, bg=COLOR_BG)
        footer.pack(side="bottom", fill="x", padx=20, pady=14)

        self.pause_button = self._make_button(footer, "Pause", self._toggle_pause, width=9)
        self.pause_button.pack(side="left")
        restart_button = self._make_button(footer, "Restart", self._restart, width=9)
        restart_button.pack(side="left", padx=(10, 0))
        menu_button = self._make_button(footer, "Menu", self._open_menu, width=7)
        menu_button.pack(side="left", padx=(10, 0))

        self.sound_button = self._make_button(footer, self._sound_label(), self._toggle_sound, width=11)
        self.sound_button.pack(side="right")
        if winsound is None:
            self.sound_button.config(state="disabled")

        # The switch that hands the controls over to the Laya brain.
        self.laya_button = self._make_button(footer, self._laya_button_label(), self._toggle_laya, width=14)
        self.laya_button.pack(side="right", padx=(0, 10))
        if self.brain is None:
            self.laya_button.config(state="disabled")

        board_holder = tk.Frame(self, bg=COLOR_BG)
        board_holder.pack(side="top", expand=True)

        self.canvas = tk.Canvas(
            board_holder,
            width=BOARD_WIDTH,
            height=BOARD_HEIGHT,
            bg=COLOR_BOARD,
            highlightthickness=2,
            highlightbackground=COLOR_BORDER,
            bd=0,
        )
        self.canvas.pack()

    def _center_window(self):
        """Place the window in the middle of the screen."""
        self.update_idletasks()
        x = max(0, (self.winfo_screenwidth() - WINDOW_WIDTH) // 2)
        y = max(0, (self.winfo_screenheight() - WINDOW_HEIGHT) // 2 - 20)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

    # -- drawing helpers -----------------------------------------------------
    def _rounded_rect(self, x1, y1, x2, y2, radius, **options):
        """Rectangle with rounded corners: a smoothed 8-point polygon."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2 - radius,
            x1, y1 + radius,
        ]
        return self.canvas.create_polygon(points, smooth=True, **options)

    @staticmethod
    def _blend(color_a, color_b, ratio):
        """Linear interpolation between two ``#rrggbb`` colors."""
        ratio = max(0.0, min(1.0, ratio))
        a = tuple(int(color_a[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(color_b[i:i + 2], 16) for i in (1, 3, 5))
        mixed = tuple(round(a[i] + (b[i] - a[i]) * ratio) for i in range(3))
        return "#%02x%02x%02x" % mixed

    def _draw_board(self):
        """Static background: the board itself plus faint grid lines."""
        self.canvas.delete("board")
        self.canvas.create_rectangle(
            0, 0, BOARD_WIDTH, BOARD_HEIGHT,
            fill=COLOR_BOARD, outline=COLOR_BORDER, width=2, tags="board",
        )
        for col in range(1, GRID_COLS):
            x = col * CELL_SIZE
            self.canvas.create_line(x, 0, x, BOARD_HEIGHT, fill=COLOR_GRID, tags="board")
        for row in range(1, GRID_ROWS):
            y = row * CELL_SIZE
            self.canvas.create_line(0, y, BOARD_WIDTH, y, fill=COLOR_GRID, tags="board")

    def _segment(self, col, row, color, inset):
        """Draw one body segment as a rounded, grid-aligned block."""
        x1 = col * CELL_SIZE + inset
        y1 = row * CELL_SIZE + inset
        x2 = (col + 1) * CELL_SIZE - inset
        y2 = (row + 1) * CELL_SIZE - inset
        self._rounded_rect(x1, y1, x2, y2, radius=(x2 - x1) * 0.3,
                           fill=color, outline="", tags="dynamic")

    def _draw_snake(self):
        segments = list(self.game.snake)
        total = len(segments)

        # Tail first so the head is drawn on top; color fades towards the tail.
        for index in range(total - 1, 0, -1):
            col, row = segments[index]
            ratio = (index - 1) / max(1, total - 2)
            color = self._blend(COLOR_SNAKE_HEAD, COLOR_SNAKE_TAIL, 0.35 + 0.65 * ratio)
            self._segment(col, row, color, inset=3)

        head_col, head_row = segments[0]
        self._segment(head_col, head_row, COLOR_SNAKE_HEAD, inset=2)
        self._draw_eyes(head_col, head_row)

    def _draw_eyes(self, col, row):
        """Two little eyes that look in the direction of travel."""
        delta_col, delta_row = self.game.direction
        center_x = col * CELL_SIZE + CELL_SIZE / 2
        center_y = row * CELL_SIZE + CELL_SIZE / 2
        forward = CELL_SIZE * 0.18
        side = CELL_SIZE * 0.20
        eye_radius = CELL_SIZE * 0.10

        for sign in (-1, 1):
            eye_x = center_x + delta_col * forward + (-delta_row) * side * sign
            eye_y = center_y + delta_row * forward + delta_col * side * sign
            self.canvas.create_oval(
                eye_x - eye_radius, eye_y - eye_radius,
                eye_x + eye_radius, eye_y + eye_radius,
                fill=COLOR_EYE, outline="", tags="dynamic",
            )

    def _draw_food(self):
        """A round apple with a shine and a leaf, clearly not a snake block."""
        if self.game.food is None:
            return
        col, row = self.game.food
        center_x = col * CELL_SIZE + CELL_SIZE / 2
        center_y = row * CELL_SIZE + CELL_SIZE / 2
        radius = CELL_SIZE * 0.33

        self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill=COLOR_FOOD, outline="", tags="dynamic",
        )
        self.canvas.create_oval(
            center_x - radius * 0.55, center_y - radius * 0.6,
            center_x - radius * 0.05, center_y - radius * 0.1,
            fill=COLOR_FOOD_SHINE, outline="", tags="dynamic",
        )
        self.canvas.create_polygon(
            center_x, center_y - radius + 1,
            center_x + radius * 0.75, center_y - radius - 3,
            center_x + radius * 0.45, center_y - radius * 0.2,
            fill=COLOR_LEAF, outline="", smooth=True, tags="dynamic",
        )

    def _draw_laya_badge(self):
        """Small tag in the corner showing the arrow the brain picked last.

        When the safety policy had to overrule that arrow, the arrow the snake
        really took is shown behind it and the tag turns gold.
        """
        if not self.laya_on or self.state != STATE_RUNNING or self._laya_direction is None:
            return
        recognized = self._laya_direction
        executed = self._laya_executed or recognized
        overruled = executed != recognized

        text = "Laya %s %s" % (LAYA_GLYPHS.get(recognized, ""), recognized)
        if overruled:
            text += " → %s" % executed

        self.canvas.create_text(
            BOARD_WIDTH - 10, 8, text=text, anchor="ne",
            font=(self.font_family, 9, "bold"),
            fill=COLOR_GOLD if overruled else COLOR_ACCENT, tags="dynamic",
        )

    def _render(self):
        """Redraw everything that moves."""
        self.canvas.delete("dynamic")
        self._draw_food()
        self._draw_snake()
        self._draw_laya_badge()
        self.canvas.tag_raise("overlay")   # keep panels above the snake

    # -- overlays (start screen, countdown, pause, game over) ----------------
    def _place_widget(self, widget, x, y):
        """Put a real Tk widget into the canvas, tracked for later cleanup."""
        self._overlay_widgets.append(widget)
        self.canvas.create_window(x, y, window=widget, tags="overlay")

    def _draw_panel(self, center_x, center_y, width, height):
        x1, y1 = center_x - width / 2, center_y - height / 2
        x2, y2 = center_x + width / 2, center_y + height / 2
        self._rounded_rect(x1, y1, x2, y2, 20,
                           fill=COLOR_PANEL, outline=COLOR_BORDER, width=2,
                           tags="overlay")

    def _clear_overlay(self):
        for widget in self._overlay_widgets:
            widget.destroy()
        self._overlay_widgets.clear()
        self._difficulty_buttons.clear()
        self.canvas.delete("overlay")

    def _show_start_screen(self):
        """Title, difficulty picker, Start button and the control hints."""
        self._clear_overlay()
        cx, cy = BOARD_WIDTH / 2, BOARD_HEIGHT / 2
        self._draw_panel(cx, cy, 430, 330)

        self.canvas.create_text(cx, cy - 118, text="SNAKE",
                                font=(self.font_family, 40, "bold"),
                                fill=COLOR_SNAKE_HEAD, tags="overlay")
        self.canvas.create_text(cx, cy - 72,
                                text="Eat the food, avoid the walls and your own tail.",
                                font=(self.font_family, 10),
                                fill=COLOR_MUTED, tags="overlay")
        self.canvas.create_text(cx, cy - 34, text="DIFFICULTY",
                                font=(self.font_family, 8, "bold"),
                                fill=COLOR_MUTED, tags="overlay")

        row = tk.Frame(self.canvas, bg=COLOR_PANEL)
        for name in DIFFICULTIES:
            button = self._make_button(row, name, lambda n=name: self._set_difficulty(n), width=8)
            button.pack(side="left", padx=4)
            self._difficulty_buttons[name] = button
        self._refresh_difficulty_buttons()
        self._place_widget(row, cx, cy + 4)

        start_button = self._make_button(self.canvas, "Start Game", self._start_new_game,
                                         width=14, bg=COLOR_ACCENT, fg="#04121f")
        self._place_widget(start_button, cx, cy + 60)

        self.canvas.create_text(cx, cy + 104,
                                text="Arrow keys / WASD to move   |   Space to pause   |   R to restart",
                                font=(self.font_family, 9),
                                fill=COLOR_MUTED, tags="overlay")
        self.canvas.create_text(cx, cy + 128,
                                text="Laya switch in the footer: let the AI play for you",
                                font=(self.font_family, 9),
                                fill=COLOR_MUTED, tags="overlay")
        self.canvas.create_text(cx, cy + 150, text="Press Enter to start",
                                font=(self.font_family, 9, "bold"),
                                fill=COLOR_ACCENT, tags="overlay")

    def _show_countdown(self, value):
        self._clear_overlay()
        cx, cy = BOARD_WIDTH / 2, BOARD_HEIGHT / 2
        self.canvas.create_text(cx, cy - 10, text=str(value),
                                font=(self.font_family, 64, "bold"),
                                fill=COLOR_ACCENT, tags="overlay")
        self.canvas.create_text(cx, cy + 62, text="Get ready!",
                                font=(self.font_family, 11),
                                fill=COLOR_MUTED, tags="overlay")

    def _show_pause_overlay(self):
        self._clear_overlay()
        cx, cy = BOARD_WIDTH / 2, BOARD_HEIGHT / 2
        self._draw_panel(cx, cy, 320, 130)
        self.canvas.create_text(cx, cy - 22, text="PAUSED",
                                font=(self.font_family, 26, "bold"),
                                fill=COLOR_ACCENT, tags="overlay")
        self.canvas.create_text(cx, cy + 24, text="Press Space or Resume to continue",
                                font=(self.font_family, 10),
                                fill=COLOR_MUTED, tags="overlay")

    def _show_game_over(self, won):
        self._clear_overlay()
        cx, cy = BOARD_WIDTH / 2, BOARD_HEIGHT / 2
        self._draw_panel(cx, cy, 380, 300)

        title = "YOU WIN!" if won else "GAME OVER"
        self.canvas.create_text(cx, cy - 104, text=title,
                                font=(self.font_family, 30, "bold"),
                                fill=COLOR_GOLD if won else COLOR_DANGER, tags="overlay")
        self.canvas.create_text(cx, cy - 50, text="Final score",
                                font=(self.font_family, 9),
                                fill=COLOR_MUTED, tags="overlay")
        self.canvas.create_text(cx, cy - 16, text=str(self.game.score),
                                font=(self.font_family, 26, "bold"),
                                fill=COLOR_TEXT, tags="overlay")
        if self.new_record:
            self.canvas.create_text(cx, cy + 20, text="New high score!",
                                    font=(self.font_family, 10, "bold"),
                                    fill=COLOR_GOLD, tags="overlay")
        else:
            self.canvas.create_text(cx, cy + 20, text="Best: %d" % self.high_score,
                                    font=(self.font_family, 10),
                                    fill=COLOR_MUTED, tags="overlay")

        row = tk.Frame(self.canvas, bg=COLOR_PANEL)
        again = self._make_button(row, "Play Again", self._restart, width=12,
                                  bg=COLOR_ACCENT, fg="#04121f")
        again.pack(side="left", padx=4)
        menu = self._make_button(row, "Menu", self._open_menu, width=8)
        menu.pack(side="left", padx=4)
        self._place_widget(row, cx, cy + 70)

        self.canvas.create_text(cx, cy + 116, text="Press Enter or R to play again",
                                font=(self.font_family, 9),
                                fill=COLOR_MUTED, tags="overlay")

    def _refresh_difficulty_buttons(self):
        """Highlight the selected difficulty button."""
        for name, button in self._difficulty_buttons.items():
            selected = name == self.difficulty
            button.config(
                bg=COLOR_ACCENT if selected else COLOR_BUTTON,
                fg="#04121f" if selected else COLOR_TEXT,
                activebackground=COLOR_ACCENT if selected else COLOR_BUTTON_ACTIVE,
                activeforeground="#04121f" if selected else COLOR_TEXT,
            )

    # -- game flow -----------------------------------------------------------
    def _start_new_game(self):
        """Reset the board and run the 3-2-1-GO countdown."""
        self._cancel_timers()
        self.game.reset()
        self._laya_direction = None
        self._laya_executed = None
        self.state = STATE_COUNTDOWN
        self._update_header()
        self._update_controls()
        self._update_laya_button()
        self._render()
        self.sound.play("start")

        self._countdown_value = COUNTDOWN_FROM
        self._show_countdown(self._countdown_value)
        self._countdown_id = self.after(COUNTDOWN_TICK_MS, self._countdown_tick)

    def _countdown_tick(self):
        self._countdown_id = None
        self._countdown_value -= 1

        if self._countdown_value > 0:
            self._show_countdown(self._countdown_value)
            self._countdown_id = self.after(COUNTDOWN_TICK_MS, self._countdown_tick)
        elif self._countdown_value == 0:
            self._show_countdown("GO!")
            self._countdown_id = self.after(COUNTDOWN_TICK_MS // 2, self._countdown_tick)
        else:
            self._clear_overlay()
            self._start_running()

    def _start_running(self):
        self.state = STATE_RUNNING
        self._update_controls()
        self._schedule_next_step()

    def _current_delay(self):
        """Milliseconds until the next step, based on difficulty and score."""
        config = DIFFICULTIES[self.difficulty]
        steps = self.game.score // config["speedup_every"]
        return max(config["min_delay"], config["delay"] - steps * config["speedup_step"])

    def _schedule_next_step(self):
        """Always cancel the old timer first, so only one loop can ever run."""
        self._cancel_loop()
        if self.laya_on and self.brain is not None:
            self._loop_id = self.after(self._current_delay(), self._laya_tick)
        else:
            self._loop_id = self.after(self._current_delay(), self._on_tick)

    def _cancel_loop(self):
        if self._loop_id is not None:
            self.after_cancel(self._loop_id)
            self._loop_id = None

    def _cancel_timers(self):
        """Cancel both the game loop and any pending countdown tick."""
        self._cancel_loop()
        if self._countdown_id is not None:
            self.after_cancel(self._countdown_id)
            self._countdown_id = None

    def _advance(self):
        """Run one game step; returns True while the round is still going."""
        event = self.game.step()

        if event == "eat":
            self.sound.play("eat")
            self._update_header()
        elif event in ("wall", "self"):
            self._end_game(won=False)
        elif event == "win":
            self._end_game(won=True)

        self._render()
        return self.state == STATE_RUNNING

    def _on_tick(self):
        """One game step, scheduled with Tk's after() - no while loop."""
        self._loop_id = None
        if self.state != STATE_RUNNING:
            return

        if self._advance():
            self._loop_id = self.after(self._current_delay(), self._on_tick)

    # -- Laya auto-pilot -----------------------------------------------------
    def _board_key(self):
        """Fingerprint of the board, so answers can be matched to a position."""
        return (tuple(self.game.snake), self.game.food, self.game.direction)

    def _board_snapshot(self):
        """Plain data for the brain - no Tk objects ever leave this thread."""
        return {
            "cols": self.game.cols,
            "rows": self.game.rows,
            "snake": list(self.game.snake),
            "food": self.game.food,
            "direction": self.game.direction,
        }

    def _laya_tick(self):
        """One Laya-steered step, played at the normal difficulty pace.

        The brain is asked about every board, and its newest arrow is followed
        as a heading until a fresher one arrives.  The snake never stops to
        wait for the model: while Laya is thinking, the safety policy drives.
        """
        self._loop_id = None
        if self.state != STATE_RUNNING or not self.laya_on or self.brain is None:
            return

        self._update_laya_button()

        board = self._board_snapshot()
        answer = self.brain.take_answer(self._board_key())
        if answer is None:
            # No answer for this exact board - follow the newest heading from
            # the boards Laya has already seen.
            answer = self.brain.take_latest()
        if answer is not None:
            self._laya_direction = answer

        executed = self.brain.resolve(board, self._laya_direction)
        self._laya_executed = executed

        self.game.turn(executed)           # this is the "pressing the arrow" part
        if not self._advance():
            return

        # Hand the fresh board over right away, so the model can think ahead.
        self.brain.submit(self._board_key(), self._board_snapshot())
        self._loop_id = self.after(self._current_delay(), self._laya_tick)

    def _end_game(self, won):
        """Stop the snake and show the result."""
        self._cancel_timers()              # the snake stops immediately
        self.state = STATE_WON if won else STATE_GAME_OVER

        self.new_record = self.game.score > self.high_score
        if self.new_record:
            self.high_score = self.game.score

        self.sound.play("game_over")
        self._update_header()
        self._update_controls()
        self._render()
        self._show_game_over(won)

    def _restart(self):
        """Start a fresh round from any state."""
        self._start_new_game()

    def _open_menu(self):
        """Back to the start screen (also the way to change difficulty)."""
        self._cancel_timers()
        self.state = STATE_START
        self._update_controls()
        self._render()
        self._show_start_screen()
        self.canvas.focus_set()

    def _toggle_pause(self):
        if self.state == STATE_RUNNING:
            self.state = STATE_PAUSED
            self._cancel_loop()            # no steps while paused
            self._update_controls()
            self._show_pause_overlay()
        elif self.state == STATE_PAUSED:
            self.state = STATE_RUNNING
            self._update_controls()
            self._clear_overlay()
            self._schedule_next_step()     # exactly one loop, never two

    def _set_difficulty(self, name):
        self.difficulty = name
        self.mode_var.set(name)
        self._refresh_difficulty_buttons()
        self.sound.play("click")

    def _toggle_sound(self):
        enabled = self.sound.toggle()
        self.sound_button.config(text=self._sound_label())
        if enabled:
            self.sound.play("click")

    def _sound_label(self):
        if winsound is None:
            return "Sound: N/A"
        return "Sound: On" if self.sound.enabled else "Sound: Off"

    # -- Laya switch ---------------------------------------------------------
    def _toggle_laya(self):
        """Hand the arrow keys over to the brain, or take them back."""
        if self.brain is None:
            return
        self.laya_on = not self.laya_on
        self.sound.play("click")

        if self.laya_on:
            self.brain.start()             # loads the model in the background
            self._laya_direction = None
            self._laya_executed = None
        self._update_laya_button()

        if self.state == STATE_RUNNING:
            self._schedule_next_step()     # swap between the two loops
        else:
            self._render()

    def _laya_button_label(self):
        if self.brain is None:
            return "Laya: N/A"
        if not self.laya_on:
            return "Laya: Off"
        status = self.brain.status
        if status == "ready":
            return "Laya: On"
        if status == "fallback":
            return "Laya: Fallback"
        return "Laya: Loading"

    def _update_laya_button(self):
        label = self._laya_button_label()
        if label == self._laya_button_text:
            return
        self._laya_button_text = label
        self.laya_button.config(
            text=label,
            bg=COLOR_ACCENT if self.laya_on else COLOR_BUTTON,
            fg="#04121f" if self.laya_on else COLOR_TEXT,
            activebackground=COLOR_ACCENT if self.laya_on else COLOR_BUTTON_ACTIVE,
            activeforeground="#04121f" if self.laya_on else COLOR_TEXT,
        )

    # -- header / buttons ----------------------------------------------------
    def _update_header(self):
        self.score_var.set(str(self.game.score))
        self.high_var.set(str(self.high_score))
        config = DIFFICULTIES[self.difficulty]
        self.speed_var.set(str(1 + self.game.score // config["speedup_every"]))
        self.mode_var.set(self.difficulty)

    def _update_controls(self):
        if self.state in (STATE_RUNNING, STATE_PAUSED):
            label = "Resume" if self.state == STATE_PAUSED else "Pause"
            self.pause_button.config(state="normal", text=label)
        else:
            self.pause_button.config(state="disabled", text="Pause")

    # -- input ---------------------------------------------------------------
    def _on_key(self, event):
        key = event.keysym.lower()

        if key in KEY_TO_DIRECTION:
            # While Laya is driving, only the brain gets to steer the snake.
            if self.state == STATE_RUNNING and not self.laya_on:
                self.game.turn(KEY_TO_DIRECTION[key])
            return "break"

        if key == "space":
            self._toggle_pause()
            return "break"

        if key in ("return", "kp_enter"):
            self._on_enter()
            return "break"

        if key == "r":
            self._restart()
            return "break"

        return None

    def _on_enter(self):
        if self.state in (STATE_START, STATE_GAME_OVER, STATE_WON):
            self._start_new_game()
        elif self.state == STATE_PAUSED:
            self._toggle_pause()

    def _on_close(self):
        self._cancel_timers()
        if self.brain is not None:
            self.brain.stop()
        self.destroy()


# ---------------------------------------------------------------------------

def main():
    SnakeApp().mainloop()


if __name__ == "__main__":
    main()
