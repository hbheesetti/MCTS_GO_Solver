import math
import random

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def ucb_score(parent_visits, child_visits, child_value, exploration_weight=1.4):
    if child_visits == 0:
        return float('inf')
    return child_value / child_visits + exploration_weight * math.sqrt(math.log(parent_visits) / child_visits)

def select(node):
    x = node.board.is_terminal()
    empty_points = node.board.get_empty_points()
    while not x:
        if len(node.children) < len(empty_points):
            return expand(node)
        else:
            node = best_child(node)
    print(node)
    return node

def expand(node):
    legal_moves = node.board.get_empty_points()
    untried_moves = [move for move in legal_moves if move not in [child.board.last_move for child in node.children]]
    selected_move = random.choice(untried_moves)
    new_board = node.board.play_move(selected_move, node.board.current_player)
    new_child = Node(new_board, parent=node)
    node.children.append(new_child)
    return new_child

def best_child(node):
    parent_visits = node.visits
    best_child = max(node.children, key=lambda child: ucb_score(parent_visits, child.visits, child.value))
    return best_child

def simulate(node):
    board = node.board
    current_board = board.copy()
    while not current_board.is_terminal():
        legal_moves = current_board.get_empty_points()
        random_move = random.choice(legal_moves)
        current_board = current_board.play_move(random_move)
    return current_board.get_result()

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

def mcts(initial_board, num_iterations):
    print(initial_board)
    root = Node(initial_board)
    print(root)

    for _ in range(num_iterations):
        selected_node = select(root)
        simulation_result = simulate(selected_node)
        backpropagate(selected_node, simulation_result)

    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.board.last_move
