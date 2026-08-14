from node import Node

def backpropagate(node: Node, reward: float) -> None:
    """Walk form 'node' up to the root, updating visit_count and total_value
    at every node along the path. The rewar'ds sign flips at each step, 
    since each parent represents the OPPONENT's turn relative to the child."""
    while node is not None:
        node.visit_count +=1
        node.total_value += reward
        reward = -reward
        node = node.parent
