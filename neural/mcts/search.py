from typing import Callable
from mcts.node import Node
from mcts.selection import select
from mcts.expansion import expand
from mcts.backpropagation import backpropagate

def mcts_search(
    root_state, 
    network_fn: Callable,
    num_simulations: int,
    ) -> tuple[int, dict]:
    """Run Alphazero-style MCTS from root_state. 

    Each simulation:
        1. SELECT - walk the tree with PUCT until an unexpanded or terminal node. 
        2. EXPAND - call network once: get (policy_probs, value), populate all children. 
        3. BACKPROP - propagate value up the tree (no rollout). 


    ARGS:
        root_state: Starting GameState
        network_fn: Callable(state) -> (policy_probs, value). 
        num_similations: number of MCTS simualtions to run. 


    Returns:
        best_action: The aciton with the highest visit count at root. 
        pi: Visit-count distribution over root's children
            {action: probability}, Used as policy target for training.
    """

    root = Node(state=root_state)

    # Expand root immediately so selection has children to traverse. 
    expand(root, network_fn)

    for _ in range(num_simulations):
        leaf = select(root)

        if leaf.is_terminal():
            value = leaf.state.reward()

        else:
            value = expand(leaf, network_fn)

        backpropagate(leaf, value)


    total_visits = sum(child.visit_count for child in root.children.values())
    pi = {
        action: child.visit_count/total_visits
        for action, child in root.children.items()
        }


    best_action = max(root.children, key=lambda a: root.children[a].visit_count)
    return best_action, pi
