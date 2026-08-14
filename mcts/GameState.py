"""
GameState — tic-tac-toe.
Board is a flat list of 9 cells, indices 0-8:
    0 1 2
    3 4 5
    6 7 8
Each cell is 'X', 'O', or None (empty).
"""

from dataclasses import dataclass
import copy

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


@dataclass
class GameState:
    board: list
    current_player: str  # 'X' or 'O'

    @staticmethod
    def initial():
        return GameState(board=[None] * 9, current_player="X")

    def legal_actions(self) -> list:
        return [i for i in range(9) if self.board[i] is None]

    def apply_action(self, action: int) -> "GameState":
        assert self.board[action] is None, f"cell {action} already occupied"
        new_board = copy.deepcopy(self.board)
        new_board[action] = self.current_player
        next_player = "O" if self.current_player == "X" else "X"
        return GameState(board=new_board, current_player=next_player)

    def _winner(self):
        for a, b, c in WIN_LINES:
            if self.board[a] is not None and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_terminal(self) -> bool:
        return self._winner() is not None or all(cell is not None for cell in self.board)

    def reward(self) -> float:
        """Reward from the perspective of self.current_player (whoever's turn
        it would be next, if the game weren't over). Since turns alternate,
        a winner is ALWAYS the player who just moved — never current_player.
        So from current_player's perspective: a winner means they lost (-1),
        no winner means a draw (0). There's no +1 case here by construction —
        rollout() re-projects this onto the perspective it actually needs."""
        winner = self._winner()
        return -1.0 if winner is not None else 0.0

    def render(self) -> str:
        symbols = [c if c is not None else "." for c in self.board]
        rows = [" ".join(symbols[i:i + 3]) for i in (0, 3, 6)]
        return "\n".join(rows)
