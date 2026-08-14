import math
from mcts.node import Node

C_PUCT = 1.5 # exploration constant; higher = more prior-guided exploration

def puct_score(parent: Node, child: Node) -> float:
    """PUCT score used by Alphazero

    Q: Average value from child's persepctive( negated, since parent is opponent).
    U: exploration bonus weighted by the policy network's prior probability.
    """

    Q = -(child.total_value/ child.visit_count) if child.visit_count > 0 else 0.0
    
    U = C_PUCT * child.prior_prob * math.sqrt(parent.visit_count) / ( 1 + child.visit_count)
    return Q + U
