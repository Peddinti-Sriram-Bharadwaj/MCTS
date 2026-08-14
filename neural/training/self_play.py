"""
Self-play: run one full game using MCTS, collect (state_array, pi, z) triples.
"""

from typing import Callable
import jax.numpy as jnp
from games.base import GameState
from mcts.search import mcts_search


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
        best_action, pi = mcts_search(state, network_fn, num_simulations)

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
