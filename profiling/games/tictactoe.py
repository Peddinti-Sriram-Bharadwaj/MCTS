"""
TicTacToe — concrete GameState implementation.
Board is a flat list of 9 cells, indices 0-8:
    0 1 2
    3 4 5
    6 7 8
Each cell is 'X', 'O', or None (empty).
"""

from dataclasses import dataclass
import copy
import jax.numpy as jnp
from games.base import GameState

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


@dataclass
class TicTacToe(GameState):
    board: list
    _current_player: str  # 'X' or 'O'

    @property
    def current_player(self) -> str:
        return self._current_player

    @staticmethod
    def initial():
        return TicTacToe(board=[None] * 9, _current_player="X")

    def legal_actions(self) -> list:
        return [i for i in range(9) if self.board[i] is None]

    def apply_action(self, action: int) -> "TicTacToe":
        assert self.board[action] is None, f"cell {action} already occupied"
        new_board = copy.deepcopy(self.board)
        new_board[action] = self.current_player
        next_player = "O" if self.current_player == "X" else "X"
        return TicTacToe(board=new_board, _current_player=next_player)

    def _winner(self):
        for a, b, c in WIN_LINES:
            if self.board[a] is not None and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_terminal(self) -> bool:
        return self._winner() is not None or all(cell is not None for cell in self.board)

    def reward(self) -> float:
        """Reward from current_player's perspective. Since turns alternate,
        a winner is always the player who just moved (never current_player).
        So: winner exists -> -1 (current player lost), no winner -> 0 (draw)."""
        winner = self._winner()
        return -1.0 if winner is not None else 0.0

    def render(self) -> str:
        symbols = [c if c is not None else "." for c in self.board]
        rows = [" ".join(symbols[i:i + 3]) for i in (0, 3, 6)]
        return "\n".join(rows)

    # ── Neural network interface ────────────────────────────────────────────

    @property
    def num_actions(self) -> int:
        return 9

    def to_array(self) -> jnp.ndarray:
        """Encode board as two binary planes from current player's perspective.
        Shape: (2, 3, 3)
          plane 0: 1 where current player has a piece
          plane 1: 1 where opponent has a piece
        """
        opponent = "O" if self.current_player == "X" else "X"
        mine     = [1.0 if c == self.current_player else 0.0 for c in self.board]
        theirs   = [1.0 if c == opponent             else 0.0 for c in self.board]
        return jnp.array([mine, theirs], dtype=jnp.float32).reshape(2, 3, 3)

