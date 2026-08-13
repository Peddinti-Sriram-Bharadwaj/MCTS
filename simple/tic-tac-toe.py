import math 
import random
from copy import deepcopy 
from collections import defaultdict

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

class TicTacToe:
    def __init__(self):
        self.board = [0]*9
        self.turn = 1


    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board[:]
        new_game.turn = self.turn
        return new_game

    def get_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def play(self, move):
        self.board[move] = self.turn
        self.turn *= -1
    
    def is_terminal(self):
        return self._get_winner() is not None or len(self.get_moves()) == 0

    def _get_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], 
            [0, 3, 6], [1, 4, 7], [2, 5, 6], 
            [0, 4, 8], [2, 4, 6]
            ]

        for line in lines:
            if self.board[line[0]] != 0 and self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
                return self.board[line[0]]
        return None

    
    def get_value(self):
        winner = self._get_winner()
        if winner == self.turn:
            return 1.0
        elif winner == -self.turn:
            return -1.0
        else:
            return 0.0


class MCTS:
    def __init__(self, c=math.sqrt(2)):
        self.c = c
        self.root = None

    def search(self, game, num_simulations):
        self.root = MCTSNode(game.copy())

        for _ in range(num_simulations):
            self._iteration(game.copy())

        best_move = max(self.root.children.keys(), 
        key = lambda m: self.root.children[m].visits)

        return best_move

    
    def _iteration(self, game):
        node = self.root
        path = [node]

        while node.children and not game.is_terminal():
            move = self._uct_select(node)
            game.play(move)
            node = node.children[move]
            path.append(node)

        if not game.is_terminal():
            for move in game.get_moves():
                node.children[move] = MCTSNode(game.copy(), parent=node)
            
            if node.children:
                move = random.choice(list(node.children.keys()))
                game.play(move)
                node = node.children[move]
                path.append(node)

            
        sim_game = game.copy()
        while not sim_game.is_terminal():
            move = random.choice(sim_game.get_moves())
            sim_game.play(move)

        
        reward = sim_game.get_value()
        for path_node in reversed(path):
            path_node.visits += 1
            path_node.value += reward
            reward *= -1

    def _uct_select(self, node):
        best_move = None
        best_uct = -float('inf')

        for move, child in node.children.items():
            exploitation = child.value / (child.visits + 1e-10)
            exploration = self.c * math.sqrt(math.log(node.visits+1)/(child.visits + 1))
            uct = exploration + exploitation

            if uct > best_uct:
                best_uct = uct
                best_move = move
        return best_move


def play_game(mcts_sims=1000):
    game = TicTacToe()
    mcts = MCTS()

    while not game.is_terminal():
        move = mcts.search(game.copy(), num_simulations=mcts_sims)
        game.play(move)
        print(f'Played move {move}, turn now: {-game.turn}')
        print_board(game)

    winner = game._get_winner()
    if winner ==1:
        print("X wins!")
    elif winner == -1:
        print("0 wins!")
    else:
        print("Draw!")

def print_board(game):
    symbols = {0: '.', 1: 'X', -1: 'O'}
    for i in range(3):
        print(' '.join(symbols[game.board[3*i + j]] for j in range(3)))
    print()

if __name__ == '__main__':
    play_game(mcts_sims=500)
