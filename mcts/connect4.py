"""
Connect-4 — concrete GameState implementation.
7 columns × 6 rows. Drop pieces from the top (specify column).
First to 4 in a row (horizontal, vertical, or diagonal) wins.
"""

from dataclasses import dataclass
import copy
from game_state_abstract import GameState


@dataclass
class Connect4(GameState):
    board: list  # 7×6 flattened: board[row * 7 + col]
    _current_player: str  # 'X' or 'O'

    ROWS = 6
    COLS = 7

    @property
    def current_player(self) -> str:
        return self._current_player

    @staticmethod
    def initial():
        return Connect4(board=[None] * (Connect4.ROWS * Connect4.COLS), _current_player="X")

    def _idx(self, row: int, col: int) -> int:
        """Convert (row, col) to flat index."""
        return row * self.COLS + col

    def legal_actions(self) -> list:
        """A column is playable if its top cell is empty."""
        return [col for col in range(self.COLS) if self.board[col] is None]

    def apply_action(self, action: int) -> "Connect4":
        """action is a column (0-6). Find the lowest empty row in that column,
        place a piece, and return a new Connect4 state."""
        assert action in self.legal_actions(), f"column {action} is full"
        new_board = copy.deepcopy(self.board)
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[self._idx(row, action)] is None:
                new_board[self._idx(row, action)] = self.current_player
                break
        next_player = "O" if self.current_player == "X" else "X"
        return Connect4(board=new_board, _current_player=next_player)

    def _check_winner(self) -> str or None:
        """Check all directions for 4 in a row. Return 'X', 'O', or None."""
        for row in range(self.ROWS):
            for col in range(self.COLS):
                player = self.board[self._idx(row, col)]
                if player is None:
                    continue
                # Check right
                if col + 3 < self.COLS and all(
                    self.board[self._idx(row, col + i)] == player for i in range(4)
                ):
                    return player
                # Check down
                if row + 3 < self.ROWS and all(
                    self.board[self._idx(row + i, col)] == player for i in range(4)
                ):
                    return player
                # Check diag down-right
                if row + 3 < self.ROWS and col + 3 < self.COLS and all(
                    self.board[self._idx(row + i, col + i)] == player for i in range(4)
                ):
                    return player
                # Check diag down-left
                if row + 3 < self.ROWS and col - 3 >= 0 and all(
                    self.board[self._idx(row + i, col - i)] == player for i in range(4)
                ):
                    return player
        return None

    def is_terminal(self) -> bool:
        """Game ends if someone won or all columns are full."""
        return self._check_winner() is not None or all(
            self.board[col] is not None for col in range(self.COLS)
        )

    def reward(self) -> float:
        """Same convention as TicTacToe: winner is always the previous player,
        so -1 if anyone won, 0 for draw."""
        winner = self._check_winner()
        return -1.0 if winner is not None else 0.0

    def render(self) -> str:
        """Pretty-print the board."""
        lines = []
        for row in range(self.ROWS):
            cells = []
            for col in range(self.COLS):
                cell = self.board[self._idx(row, col)]
                cells.append(cell if cell is not None else ".")
            lines.append(" ".join(cells))
        # Add column numbers at the bottom
        lines.append(" ".join(str(i) for i in range(self.COLS)))
        return "\n".join(lines)
