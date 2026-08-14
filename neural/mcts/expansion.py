from typing import Callable
import jax.numpy as jnp
from mcts.node import Node

def expand(node: Node, network_fn: Callable) -> float:
    """Evaluate 'node' with the network, expand all legal children with their prior
    probabilities, and return the value estimate for backpropagation. 

    Args:
        node: The leaf ndoe to expand. Must not already be expanded. 
        network_fn: Callable (state) -> (policy_probs, value).
                    policy_probls is a jnp array of shape (num_actions,).
                    value is a flaot in [-1, 1]


    Returns:
        value: The network's value estimate from the current player's perspective. 
    """

    policy_probs, value = network_fn(node.state)

    legal = node.state.legal_actions()

    # Mask and renormalize: zero out illegal aciotns, rescale to sum to 1. 
    mask = jnp.zeros(node.state.num_actions).at[jnp.array(legal)].set(1.0)
    masked_probs = policy_probs * mask
    total = masked_probs.sum()
    if total > 0:
        masked_probs = masked_probs / total

    else:
# Fallback: uniform over legal actions ( shouldn't happen with a trianed net)
        masked_probs = mask / len(legal)


    for action in legal:
        next_state = node.state.apply_action(action)
        child=Node(
        state=next_state,
        parent=node, 
        action_taken=action, 
        prior_prob=float(masked_probs[action]),
        )
        node.children[action] = child

    return float(value)

