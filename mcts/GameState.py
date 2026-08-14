from dataclasses import dataclass, field
from typing import Optional
import copy

@dataclass
class GameState:
    board: list
    current_player: str

    def legal_actions(self) -> list[int]:
        """which cell indices (0-8) are emtpy and playable."""
        ...

    def apply_action(self, action: int) -> "GameState":
        """Returns a NEW GameState with the move applied. Does not mutate self."""
        ...

    def is_terminal(self) -> bool:
        """True if someone won or the board is full."""
        ...

    def reward(self) -> float:
        """+1 if current player's opponent just own. -1 if current player just won,
        0 for a draw, only meaningful when is_terminal() is True."""
        ...                
