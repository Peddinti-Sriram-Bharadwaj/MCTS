"""
Inference wrapper: GameState -> (policy_probs, value).
This is what gets passed as network_fn to mcts_search and expand.
"""

import jax
import jax.numpy as jnp
from games.base import GameState
from network.model import AlphaZeroNet


def make_network_fn(model: AlphaZeroNet):
    """Return a closure that captures the model and acts as network_fn.

    Usage:
        network_fn = make_network_fn(model)
        policy_probs, value = network_fn(state)

    The returned function is what you pass to mcts_search() and expand().
    """

    @jax.jit
    def _forward(x: jnp.ndarray):
        return model(x)

    def network_fn(state: GameState) -> tuple[jnp.ndarray, float]:
        # Encode state and flatten for the MLP
        x = state.to_array().flatten()

        policy_logits, value = _forward(x)

        # Mask illegal actions: set logits of illegal moves to -inf
        legal = state.legal_actions()
        mask = jnp.full(state.num_actions, -jnp.inf)
        mask = mask.at[jnp.array(legal)].set(0.0)
        masked_logits = policy_logits + mask

        policy_probs = jax.nn.softmax(masked_logits)

        return policy_probs, float(value[0])

    return network_fn
