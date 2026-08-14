import random
import math
from typing import Callable, Optional
import jax.numpy as jnp
import numpy as np

from mcts.node import Node
from mcts.selection import select
from mcts.expansion import expand
from mcts.backpropagation import backpropagate


def _add_dirichlet_noise(node: Node, alpha: float, epsilon: float) -> None:
    """Mix Dirichlet noise into root children's priors.

    Prevents the search from collapsing onto the network's initial favourite
    move, forcing exploration of alternatives even when the network is confident.

    Args:
        node:    Root node (must already be expanded).
        alpha:   Dirichlet concentration parameter. Smaller = more noise.
                 AlphaZero uses 10/num_actions. For TicTacToe: 10/9 ≈ 1.11.
        epsilon: Mixing weight. AlphaZero uses 0.25.
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for action, eta in zip(actions, noise):
        child = node.children[action]
        child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * eta


def _sample_action(pi: dict, temperature: float) -> int:
    """Sample an action from the visit-count distribution pi.

    temperature=1 → sample proportionally to visit counts (diverse training data)
    temperature=0 → argmax (deterministic, used at evaluation time)
    """
    if temperature == 0:
        return max(pi, key=pi.get)

    actions = list(pi.keys())
    probs   = np.array([pi[a] for a in actions], dtype=np.float64)

    # Apply temperature: p_i^(1/T), then renormalise
    probs = np.power(probs, 1.0 / temperature)
    probs /= probs.sum()

    return int(np.random.choice(actions, p=probs))


def mcts_search(
    root_state,
    network_fn: Callable,
    num_simulations: int,
    temperature: float = 0.0,
    dirichlet_alpha: Optional[float] = None,
    dirichlet_epsilon: float = 0.25,
) -> tuple[int, dict]:
    """Run AlphaZero-style MCTS from root_state.

    Each simulation:
        1. SELECT  — walk the tree with PUCT until an unexpanded or terminal node.
        2. EXPAND  — call network once: get (policy_probs, value), populate all children.
        3. BACKPROP — propagate value up the tree (no rollout).

    Args:
        root_state:        Starting GameState.
        network_fn:        Callable (state) -> (policy_probs, value).
        num_simulations:   Number of MCTS simulations to run.
        temperature:       Move selection temperature.
                           1.0 = sample proportionally (training).
                           0.0 = argmax (evaluation).
        dirichlet_alpha:   Dirichlet concentration for root noise.
                           None = no noise (evaluation mode).
                           Recommended: 10 / num_actions for training.
        dirichlet_epsilon: Weight of Dirichlet noise vs network prior (default 0.25).

    Returns:
        best_action: Selected action (argmax or sampled depending on temperature).
        pi:          Visit-count distribution {action: probability}.
                     Used as policy target for training.
    """
    root = Node(state=root_state)

    # Expand root immediately so selection has children to traverse.
    expand(root, network_fn)

    # Add exploration noise at root during training.
    if dirichlet_alpha is not None and root.is_expanded():
        _add_dirichlet_noise(root, alpha=dirichlet_alpha, epsilon=dirichlet_epsilon)

    for _ in range(num_simulations):
        leaf = select(root)

        if leaf.is_terminal():
            value = leaf.state.reward()
        else:
            value = expand(leaf, network_fn)

        backpropagate(leaf, value)

    # Build visit-count distribution (pi) over root's children.
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits > 0:
        pi = {
            action: child.visit_count / total_visits
            for action, child in root.children.items()
        }
    else:
        pi = {
            action: child.prior_prob
            for action, child in root.children.items()
        }

    best_action = _sample_action(pi, temperature)
    return best_action, pi
