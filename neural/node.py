from typing import Optional
from game_state_abstract import GameState

class Node:
    def __init__(self, state: GameState, parent: Optional["Node"] = None, action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: dict[int, "Node"] = {}
        self.untried_actions = state.legal_actions()
        self.visit_count = 0
        self.total_value = 0.0


    def is_fully_expanded(self)-> bool:
        return len(self. untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

