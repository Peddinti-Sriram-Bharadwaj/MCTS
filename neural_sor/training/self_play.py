"""
Self-play: run one full game using MCTS, collect (state_array, pi, max_q) triples.

TD-SOR uses the immediate maximum Q-value from the MCTS as the target for the
neural network, rather than waiting for the final game outcome.
"""

from typing import Callable
import jax.numpy as jnp
from games.base import GameState
from mcts.search import mcts_search

DIRICHLET_ALPHA   = 1.11
DIRICHLET_EPSILON = 0.25
TEMPERATURE = 1.0


def self_play_game(
    initial_state: GameState,
    network_fn: Callable,
    num_simulations: int = 100,
) -> list[tuple[jnp.ndarray, dict, float]]:
    """Play one game of self-play from initial_state to terminal.

    Returns:
        A list of (state_array, pi, max_q) triples, one per move.
        state_array: encoded board at that step, shape (input_size,).
        pi:          MCTS visit-count distribution {action: prob}.
        max_q:       Maximum Q-value from the root node (TD Target).
    """
    state = initial_state
    trajectory = []

    while not state.is_terminal():
        best_action, pi, max_q = mcts_search(
            state,
            network_fn,
            num_simulations,
            temperature=TEMPERATURE,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
        )

        trajectory.append((
            state.to_array().flatten(),
            pi,
            max_q,  # TD-SOR uses the MCTS Q-value as the target immediately
        ))

        state = state.apply_action(best_action)

    # In TD learning, we don't back-fill the terminal reward. 
    # Each state already captured its Bellman target (max_q).
    return trajectory
