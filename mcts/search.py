from node import Node
from selection import select
from expansion import expand
from simulation import rollout
from backprop import backpropagate

def mcts_search(root_state, num_iterations: int) -> int:
    root = Node(state=root_state)

    for _ in range(num_iterations):
        leaf = select(root)

        if not leaf.is_terminal():
            leaf = expand(leaf)

        reward = rollout(leaf.state)
        backpropagate(leaf, reward)


    best_action = max(root.children, key=lambda action: root.children[action].visit_count)
    return best_action
