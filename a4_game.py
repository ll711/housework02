"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
from __future__ import annotations
#main game loop
# load a1, a2, a3 files and give a GUI graphical interface to choose which moode to
# run(matichine VS machine, human VS machine)
#in the GUI, show a star path, and import the path from a3 file
# show the path on the GUI and give who need to active ahead
"""
This part is the building-block assembly stage.
Put all the completed modules into it.
And finally complete the delivery.
"""

#main game loop
# load a1, a2, a3 files and give a GUI graphical interface to choose which moode to
# run(matichine VS machine, human VS machine)
#in the GUI, show a star path, and import the path from a3 file
# show the path on the GUI and give who need to active ahead

from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import builtins

# --- A1 / A3 Dependency ---
from a1_state import State
builtins.state = State  # Compatible with the usage of lowercase "state" in certain codes

import a3_agent as a3
a3.state = State
from a3_agent import Agent

Coord = Tuple[int, int]
CELL = 36
PAD  = 14
AI_CHOICES = ["AlphaBeta", "MiniMax"]


class GameApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hinger – A4 Game")

        # --- ask board size ---
        rows = simpledialog.askinteger("Board", "Rows m (≥2):", minvalue=2, initialvalue=8, parent=self.root)
        cols = simpledialog.askinteger("Board", "Cols n (≥2):", minvalue=2, initialvalue=8, parent=self.root)
        if not rows or not cols:
            rows, cols = 8, 8
        self.rows, self.cols = rows, cols

        # model
        self.grid: List[List[int]] = [[0]*self.cols for _ in range(self.rows)]
        self.state: Optional[State] = None
        self.turn: str = "Player A"     # "Player A"|"Player B"|"AI-A"|"AI-B"
        self.mode_var = tk.StringVar(value="Human vs AI")
        self.ai_a_var = tk.StringVar(value="AlphaBeta")  # AI-A logic
        self.ai_b_var = tk.StringVar(value="AlphaBeta")  # AI-B logic
        self.game_over: bool = False
        self.triggered_hinger_once: bool = False

        # store hinger snapshot for red highlight after game end
        self._hinger_coords: List[Tuple[int, int]] = []
        self._grid_snapshot: Optional[List[List[int]]] = None

        # layout
        self._build_ui()
        self._make_entry_overlays()
        self._draw_board()

        # --- Info Panel (Hinger / Region) ---
        # Placed below the main window grid (canvas occupies row=0..30, put this at row=31)
        self.frame_info = tk.Frame(self.root)
        self.frame_info.grid(row=31, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.lbl_hinger = tk.Label(self.frame_info, text="Hinger Count: 0",
                                   font=("Arial", 11, "bold"), fg="red")
        self.lbl_region = tk.Label(self.frame_info, text="Region Count: 0",
                                   font=("Arial", 11, "bold"), fg="blue")
        self.lbl_hinger.grid(row=0, column=0, padx=(8, 12))
        self.lbl_region.grid(row=0, column=1)

    # ---------- UI ----------
    def _build_ui(self):
        width  = self.cols * CELL + 2*PAD
        height = self.rows * CELL + 2*PAD
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=30)
        self.canvas.bind("<Button-1>", self.on_click)

        panel = tk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="n")

        ttk.Label(panel, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.mode_var,
                     values=["Human vs AI", "Human vs Human", "AI vs AI"],
                     state="readonly", width=14).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(panel, text="AI-A Logic").grid(row=1, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.ai_a_var,
                     values=AI_CHOICES, state="readonly", width=14).grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(panel, text="AI-B Logic").grid(row=2, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.ai_b_var,
                     values=AI_CHOICES, state="readonly", width=14).grid(row=2, column=1, padx=4, pady=2)

        self.btn_confirm = tk.Button(panel, text="Confirm Board", command=self.on_confirm_board)
        self.btn_confirm.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        self.btn_next = tk.Button(panel, text="Next Turn", command=self.on_next_turn, state="disabled")
        self.btn_next.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        self.btn_reset = tk.Button(panel, text="Reset / Re-enter", command=self.on_reset)
        self.btn_reset.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 2))

        self.lbl_status = ttk.Label(panel, text="Fill numbers then Confirm",
                                    foreground="#444", wraplength=180, justify="left")
        self.lbl_status.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _make_entry_overlays(self):
        """Place Entry widgets on top of every cell before the board is confirmed."""
        self.entries: List[List[tk.Entry]] = []
        for r in range(self.rows):
            row_e: List[tk.Entry] = []
            for c in range(self.cols):
                x = PAD + c*CELL + 2
                y = PAD + r*CELL + 2
                e = tk.Entry(self.canvas, width=3, justify="center")
                self.canvas.create_window(x, y, window=e, anchor="nw", width=CELL-4, height=CELL-4)
                row_e.append(e)
            self.entries.append(row_e)

    # ---------- Drawing ----------
    def _draw_board(self):
        self.canvas.delete("grid")
        # grid + numbers (during play)
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = PAD + c*CELL
                y1 = PAD + r*CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                v = self.grid[r][c]
                fill = "#f5f7fa" if v == 0 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#c9c9c9", tags="grid")
                if not getattr(self, "entries", None) and v > 0 and not self.game_over:
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                            text=str(v), fill="#222",
                                            font=("Arial", max(10, CELL//2)), tags="grid")

        # game over: paint all hinger cells red (with value snapshot)
        if self.game_over and self._grid_snapshot is not None:
            for (r, c) in self._hinger_coords:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    x1 = PAD + c*CELL
                    y1 = PAD + r*CELL
                    x2 = x1 + CELL
                    y2 = y1 + CELL
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ff8080", outline="#c66", tags="grid")
                    v = self._grid_snapshot[r][c]
                    if v > 0:
                        self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                                text=str(v), fill="#111",
                                                font=("Arial", max(10, CELL//2), "bold"), tags="grid")
        self.root.title(f"Hinger – Turn: {self.turn}")

        # --- Update Hinger / Region counts ---
        try:
            self._refresh_hingers()
            hinger_count = len(getattr(self.state, "true_hinger_global_coords", [])) if self.state else 0
        except Exception:
            hinger_count = 0
        region_count = self._count_regions()
        if hasattr(self, "lbl_hinger"):
            self.lbl_hinger.config(text=f"Hinger Count: {hinger_count}")
        if hasattr(self, "lbl_region"):
            self.lbl_region.config(text=f"Region Count: {region_count}")

    # ---------- Helpers ----------
    def _refresh_hingers(self):
        if self.state is None:
            return
        try:
            self.state.IS_Hinger(full_scan=True)
        except TypeError:
            self.state.IS_Hinger()

    def _is_hinger_cell(self, rc: Tuple[int, int]) -> bool:
        if self.state is None:
            return False
        coords = getattr(self.state, "true_hinger_global_coords", [])
        return rc in coords

    def _apply_minus_one(self, rc: Tuple[int, int]):
        r, c = rc
        if self.grid[r][c] <= 0:
            return False
        self.grid[r][c] -= 1
        if self.state is not None:
            self.state.result = [row[:] for row in self.grid]
            try:
                self.state.Get_Graph()
            except Exception:
                pass
        return True

    def _all_cleared(self) -> bool:
        return all(v == 0 for row in self.grid for v in row)

    def _opponent(self) -> str:
        mode = self.mode_var.get()
        if mode == "Human vs AI":
            return "AI-B" if self.turn == "Player A" else "Player A"
        if mode == "Human vs Human":
            return "Player B" if self.turn == "Player A" else "Player A"
        # AI vs AI
        return "AI-B" if self.turn == "AI-A" else "AI-A"

    def _swap_turn(self):
        self.turn = self._opponent()

    def _count_regions(self) -> int:
        """Count number of 8-neighbour connected regions of cells with value > 0"""
        rows, cols = self.rows, self.cols
        seen = [[False]*cols for _ in range(rows)]
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),( -1,1)]
        from collections import deque

        def bfs(sr, sc):
            q = deque([(sr, sc)])
            seen[sr][sc] = True
            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and self.grid[nr][nc] > 0:
                        seen[nr][nc] = True
                        q.append((nr, nc))
        count = 0
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] > 0 and not seen[r][c]:
                    bfs(r, c)
                    count += 1
        return count

    # ---------- Events ----------
    def on_confirm_board(self):
        # read entries
        for r in range(self.rows):
            for c in range(self.cols):
                txt = self.entries[r][c].get().strip()
                try:
                    v = int(txt)
                except Exception:
                    v = 0
                self.grid[r][c] = v

        # remove overlays
        for row in self.entries:
            for e in row:
                e.destroy()
        self.entries.clear()

        # init state + first hinger scan
        self.state = State([row[:] for row in self.grid])
        try:
            self.state.Get_Graph()
        except Exception:
            pass
        self._refresh_hingers()

        self.btn_next.config(state="normal")
        self.btn_confirm.config(state="disabled")
        self.game_over = False
        self.triggered_hinger_once = False
        self._hinger_coords = []
        self._grid_snapshot = None

        mode = self.mode_var.get()
        if mode == "AI vs AI":
            self.turn = "AI-A"
        else:
            self.turn = "Player A"

        self.lbl_status.config(text=f"Game started: {mode}")
        self._draw_board()

        # if it's AI to move first, move now
        if self._current_is_ai():
            self.root.after(50, self._ai_move_current)

    def on_click(self, ev):
        if self.game_over or self.state is None:
            return
        # block clicks if it's an AI turn
        if self._current_is_ai():
            return
        r = (ev.y - PAD) // CELL
        c = (ev.x - PAD) // CELL
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        self._handle_move((r, c))

    def _current_is_ai(self) -> bool:
        return self.turn in ("AI-A", "AI-B")

    def _handle_move(self, rc: Tuple[int, int]):
        if self.game_over or self.state is None:
            return

        r, c = rc
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.grid[r][c] <= 0:
            winner = self._opponent()
            self.game_over = True
            messagebox.showinfo("Game Over", f"Illegal move. {winner} wins.")
            self._draw_board()
            return

        self._refresh_hingers()

        if self._is_hinger_cell(rc):
            self._grid_snapshot = [row[:] for row in self.grid]
            self._hinger_coords = list(getattr(self.state, "true_hinger_global_coords", []))
            self._apply_minus_one(rc)
            self.triggered_hinger_once = True
            self.game_over = True
            messagebox.showinfo("Win", f"{self.turn} played on a hinger and wins!")
            self._draw_board()
            return

        self._apply_minus_one(rc)

        if self._all_cleared() and not self.triggered_hinger_once:
            self.game_over = True
            messagebox.showinfo("Draw", "All counters removed, no hinger triggered.")
            self._draw_board()
            return

        self._swap_turn()
        self._draw_board()

        # if next is AI, trigger immediately
        if self._current_is_ai() and not self.game_over:
            self.root.after(50, self._ai_move_current)

    def _ai_move_current(self):
        if self.turn == "AI-A":
            self._ai_move(which="A")
        elif self.turn == "AI-B":
            self._ai_move(which="B")

    def _ai_move(self, which: str = "B"):
        if self.game_over or self.state is None:
            return
        agent = Agent()
        if which == "A":
            agent.model = self.ai_a_var.get().lower()
        else:
            agent.model = self.ai_b_var.get().lower()
        mv = agent.move(self.state, mode=agent.model)
        if mv is None:
            self.game_over = True
            winner = self._opponent()
            messagebox.showinfo("Game Over", f"{self.turn} has no legal move. {winner} wins.")
            self._draw_board()
            return
        self._handle_move(mv)

    def on_next_turn(self):
        """Advance one step of AI.
        - If current player is an AI, just make that AI move.
        - If Human vs AI and it's human's turn, swap to the AI and make it move (acts like 'pass').
        - If AI vs AI and somehow it's not an AI's turn, set to AI-A and move.
        """
        if self.game_over or self.state is None:
            return

        # Case 1: already AI's turn
        if self._current_is_ai():
            self._ai_move_current()
            return

        mode = self.mode_var.get()
        # Case 2: Human vs AI but it's human's turn -> step to AI and move
        if mode == "Human vs AI":
            self.turn = "AI-B"
            self.lbl_status.config(text="Stepped to AI-B turn via Next Turn")
            self._draw_board()
            self._ai_move_current()
            return

        # Case 3: AI vs AI but turn not AI (shouldn't happen). Force to AI-A.
        if mode == "AI vs AI":
            self.turn = "AI-A"
            self.lbl_status.config(text="Forced AI-A to move via Next Turn")
            self._draw_board()
            self._ai_move_current()
            return

        # Human vs Human: no AI to move.
        self.lbl_status.config(text="Next Turn: Human vs Human — no AI to move.")
        return

    def on_reset(self):
        self.canvas.delete("all")
        rows = simpledialog.askinteger("Board", "Rows m (≥2):", minvalue=2, initialvalue=self.rows, parent=self.root)
        cols = simpledialog.askinteger("Board", "Cols n (≥2):", minvalue=2, initialvalue=self.cols, parent=self.root)
        if not rows or not cols:
            rows, cols = self.rows, self.cols
        self.rows, self.cols = rows, cols
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.state = None
        self.turn = "Player A"
        self.game_over = False
        self.triggered_hinger_once = False
        self._hinger_coords = []
        self._grid_snapshot = None

        width  = self.cols * CELL + 2*PAD
        height = self.rows * CELL + 2*PAD
        self.canvas.config(width=width, height=height)
        self._make_entry_overlays()
        self.btn_confirm.config(state="normal")
        self.btn_next.config(state="disabled")
        self.lbl_status.config(text="Fill numbers then Confirm")
        self._draw_board()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GameApp().run()
