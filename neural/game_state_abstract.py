"""
Abstract GameState interface — any game (tic-tac-toe, Connect-4, etc.)
must implement these methods.
"""

from abc import ABC, abstractmethod


class GameState(ABC):
    """Abstract base class for game states. Concrete games inherit from this
    and implement all abstract methods."""

    @property
    @abstractmethod
    def current_player(self) -> str:
        """Return whose turn it is ('X', 'O', or game-specific identifier)."""
        pass

    @abstractmethod
    def legal_actions(self) -> list:
        """Return a list of valid moves from this state."""
        pass

    @abstractmethod
    def apply_action(self, action) -> "GameState":
        """Return a new GameState with the action applied. Do not mutate self."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over (someone won or draw)."""
        pass

    @abstractmethod
    def reward(self) -> float:
        """Return the reward from current_player's perspective. Only meaningful
        when is_terminal() is True. Convention: negative if the previous player won."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable string representation of the board."""
        pass
