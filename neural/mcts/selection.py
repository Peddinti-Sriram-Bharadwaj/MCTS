import math
from mcts.node import Node

C_PUCT = 1.5  # exploration constant; higher = more prior-guided exploration


def puct_score(parent: Node, child: Node) -> float:
    """PUCT score used by AlphaZero.

    Q: Average value from child's perspective (negated, since parent is opponent).
    U: Exploration bonus weighted by the policy network's prior probability.
    """
    Q = -(child.total_value / child.visit_count) if child.visit_count > 0 else 0.0
    U = C_PUCT * child.prior_prob * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return Q + U


def select(node: Node) -> Node:
    """Walk from 'node' down to a leaf using PUCT scores, stopping at an
    unexpanded node or a terminal state."""
    while node.is_expanded() and not node.is_terminal():
        node = max(node.children.values(), key=lambda child: puct_score(node, child))
    return node
