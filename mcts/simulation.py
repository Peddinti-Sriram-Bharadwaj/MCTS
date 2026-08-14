import random
from gamestate import GameState

def rollout(state: GameState) -> float:
    """Play random legal moves from 'state' until the game ends, 
    then return the reward from the perspective of the player 
    who was about to move at the ORIGINAL state passed in."""

    original_player = state.current_player
    current = state

    while not current.is_terminal():
        action = random.choice(current.legal_actions())
        current = current.apply_action(action)


    reward = current.reward()
    if current.current_player != original_player:
        reward = -reward
    return reward
