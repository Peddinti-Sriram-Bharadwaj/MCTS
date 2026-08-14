"""
Run with: python3 play.py
MCTS plays both sides of tic-tac-toe against itself, printing the board
after every move. A correctly-working MCTS should always draw against
itself (tic-tac-toe is a solved game — perfect play from both sides draws).
"""

from gamestate import GameState
from search import mcts_search

ITERATIONS_PER_MOVE = 500

state = GameState.initial()
move_number = 1

while not state.is_terminal():
    action = mcts_search(state, num_iterations=ITERATIONS_PER_MOVE)
    state = state.apply_action(action)
    print(f"--- move {move_number} (played cell {action}) ---")
    print(state.render())
    print()
    move_number += 1

winner = state._winner()
if winner is not None:
    print(f"Winner: {winner}")
else:
    print("Draw.")
