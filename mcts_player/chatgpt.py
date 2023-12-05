import math
import random

from board import GoBoard

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

def uct(node):
    if node.visits == 0:
        return float('inf')
    exploitation = node.value / node.visits
    exploration = math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation + 10000 * exploration

def select(node):
    while not node.state.is_terminal() and len(node.children) > 0:
        node = max(node.children, key=uct)
    return node

def expand(node):
    actions = node.state.get_empty_points()
    for action in actions:
        child_state = node.state.play_move(action, node.state.current_player)
        child_node = Node(node.state, action=action, parent=node)
        node.children.append(child_node)
    return random.choice(node.children)

def simulate(board:GoBoard):
    while not (board.is_terminal()):
        action = random.choice(board.get_empty_points())
        board.play_move(action, board.current_player)
    return board.get_result__()

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(root_state: GoBoard, iterations):
    state = root_state.copy()
    root_node = Node(state)

    for _ in range(iterations):
        selected_node = select(root_node)
        expanded_node = expand(selected_node)
        reward = simulate(expanded_node.state)
        backpropagate(expanded_node, reward)

    best_child = max(root_node.children, key=lambda x: x.value)
    return best_child.action
