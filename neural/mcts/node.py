from typing import Optional
from games.base import GameState

class Node:
    def __init__(
    self, 
    state: GameState, 
    parent: Optional["Node"] = None, 
    action_taken: Optional[int] = None, 
    prior_prob: float = 0.0,
    ):

        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob

        self.children: dict[int, "Node"] = {}
        self.visit_count = 0
        self.total_value = 0.0


    def is_expanded(self) -> bool:
        """True once the network has evaluated this node and populated children."""
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()         
