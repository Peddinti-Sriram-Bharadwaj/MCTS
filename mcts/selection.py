import math
from node import Node

UCB_C = 1.41

def ucb_score(parent: Node, child: Node) -> float:
    if child.visit_count ==0:
        return float('inf')
    exploitation = -(child.total_value/ child.visit_count)
    exploration = UCB_C * math.sqrt(math.log(parent.visit_count)/ child.visit_count)
    return exploration + exploitation

def select(node: Node) -> Node:
""" Walk from 'node' down to a leaf, always picking the child with the
highest UCB score,stopping when we hit a node that's not fully expanded or is terminal."""

    while node.is_fully_expanded() and not node.is_terminal():
        node = max(node.children.values(), key = lambda child: ucb_score(node, child))
    return node

