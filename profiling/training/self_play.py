"""
Self-play: run one full game using MCTS, collect (state_array, pi, z) triples.

Training uses:
  - Dirichlet noise at root  → forces exploration
  - Temperature=1 sampling   → generates diverse, non-trivial positions
  - Fewer MCTS simulations   → network priors matter, imperfect play creates wins/losses
"""

from typing import Callable
import jax.numpy as jnp
from games.base import GameState
from mcts.search import mcts_search


# AlphaZero recommendation: alpha = 10 / num_actions
# TicTacToe has 9 actions → alpha ≈ 1.11
DIRICHLET_ALPHA   = 1.11
DIRICHLET_EPSILON = 0.25

# Use temperature=1 for the whole game during early training.
# In full AlphaZero this drops to 0 after move 30, but TicTacToe
# games are short (≤9 moves) so temperature=1 throughout is fine.
TEMPERATURE = 1.0


def self_play_game(
    initial_state: GameState,
    network_fn: Callable,
    num_simulations: int = 100,
) -> list[tuple[jnp.ndarray, dict, float]]:
    """Play one game of self-play from initial_state to terminal.

    Returns:
        A list of (state_array, pi, z) triples, one per move.
        state_array: encoded board at that step, shape (input_size,).
        pi:          MCTS visit-count distribution {action: prob}.
        z:           Game outcome from that move's player's perspective.
    """
    state = initial_state
    trajectory = []  # list of (state_array, pi, player_at_move)

    while not state.is_terminal():
        best_action, pi = mcts_search(
            state,
            network_fn,
            num_simulations,
            temperature=TEMPERATURE,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
        )

        # Record before applying the move
        trajectory.append((
            state.to_array().flatten(),
            pi,
            state.current_player,
        ))

        state = state.apply_action(best_action)

    # Game over: assign z to each position retroactively.
    # state.reward() returns reward from state.current_player's perspective.
    # By convention: reward < 0 means current_player lost (the previous player won).
    terminal_reward = state.reward()
    terminal_player = state.current_player  # property, not a method call

    examples = []
    for state_array, pi, player_at_move in trajectory:
        if terminal_reward == 0.0:
            z = 0.0   # draw
        elif player_at_move == terminal_player:
            z = terminal_reward         # same perspective as terminal state
        else:
            z = -terminal_reward        # flip for the other player
        examples.append((state_array, pi, z))

    return examples
