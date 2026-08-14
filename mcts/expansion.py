from node import Node

def expand(node: Node) -> Node:
    """Take one untried action from 'node', create the resulting child Node, 
    and return that new child."""
    action = node.untried_actions.pop()
    next_state = node.state.apply_action(action)
    child = Node(state=next_state, parent=node, action_taken=action)
    node.children[action] = child
    return child

