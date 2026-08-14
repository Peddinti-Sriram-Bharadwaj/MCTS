"""
Connect-4 — concrete GameState implementation.
Board is a flat list of 42 cells (6 rows x 7 cols), indices 0-41:
    0  1  2  3  4  5  6   (top row, row 0)
    ...
    35 36 37 38 39 40 41  (bottom row, row 5)
Each cell is 'X', 'O', or None (empty).
"""

from dataclasses import dataclass
import copy
import jax.numpy as jnp
from games.base import GameState

ROWS = 6
COLS = 7

# Pre-calculate all winning lines (4-in-a-row)
WIN_LINES = []
# Horizontal
for r in range(ROWS):
    for c in range(COLS - 3):
        idx = r * COLS + c
        WIN_LINES.append((idx, idx + 1, idx + 2, idx + 3))
# Vertical
for r in range(ROWS - 3):
    for c in range(COLS):
        idx = r * COLS + c
        WIN_LINES.append((idx, idx + COLS, idx + 2 * COLS, idx + 3 * COLS))
# Diagonal /
for r in range(3, ROWS):
    for c in range(COLS - 3):
        idx = r * COLS + c
        WIN_LINES.append((idx, idx - COLS + 1, idx - 2 * COLS + 2, idx - 3 * COLS + 3))
# Diagonal \
for r in range(ROWS - 3):
    for c in range(COLS - 3):
        idx = r * COLS + c
        WIN_LINES.append((idx, idx + COLS + 1, idx + 2 * COLS + 2, idx + 3 * COLS + 3))


@dataclass
class Connect4(GameState):
    board: list
    _current_player: str  # 'X' or 'O'

    @property
    def current_player(self) -> str:
        return self._current_player

    @staticmethod
    def initial():
        return Connect4(board=[None] * (ROWS * COLS), _current_player="X")

    def legal_actions(self) -> list:
        # A column is valid if its top cell is empty
        return [c for c in range(COLS) if self.board[c] is None]

    def apply_action(self, action: int) -> "Connect4":
        assert 0 <= action < COLS, f"Invalid column {action}"
        assert self.board[action] is None, f"Column {action} is full"
        
        new_board = copy.deepcopy(self.board)
        
        # Apply gravity: find the lowest empty row in this column
        for r in range(ROWS - 1, -1, -1):
            idx = r * COLS + action
            if new_board[idx] is None:
                new_board[idx] = self.current_player
                break
                
        next_player = "O" if self.current_player == "X" else "X"
        return Connect4(board=new_board, _current_player=next_player)

    def _winner(self):
        for a, b, c, d in WIN_LINES:
            if (self.board[a] is not None and 
                self.board[a] == self.board[b] == self.board[c] == self.board[d]):
                return self.board[a]
        return None

    def is_terminal(self) -> bool:
        return self._winner() is not None or all(cell is not None for cell in self.board)

    def reward(self) -> float:
        """Reward from current_player's perspective.
        Since turns alternate, the player who just moved (the opponent) is the only possible winner.
        So: winner exists -> -1 (current player lost)."""
        winner = self._winner()
        return -1.0 if winner is not None else 0.0

    def render(self) -> str:
        symbols = [c if c is not None else "." for c in self.board]
        rows_str = []
        for r in range(ROWS):
            rows_str.append(" ".join(symbols[r * COLS : (r + 1) * COLS]))
        rows_str.append("- " * COLS)
        rows_str.append(" ".join(str(i) for i in range(COLS)))
        return "\n".join(rows_str)

    # ── Neural network interface ────────────────────────────────────────────

    @property
    def num_actions(self) -> int:
        return COLS

    def to_array(self) -> jnp.ndarray:
        """Encode board as two binary planes from current player's perspective.
        Shape: (2, 6, 7)
          plane 0: 1 where current player has a piece
          plane 1: 1 where opponent has a piece
        """
        opponent = "O" if self.current_player == "X" else "X"
        mine     = [1.0 if c == self.current_player else 0.0 for c in self.board]
        theirs   = [1.0 if c == opponent             else 0.0 for c in self.board]
        return jnp.array([mine, theirs], dtype=jnp.float32).reshape(2, ROWS, COLS)
