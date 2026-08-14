"""
Run with:
    python3 play.py --game tictactoe
    python3 play.py --game connect4
"""

import argparse
from search import mcts_search
from tictactoe import TicTacToe
from connect4 import Connect4

GAMES = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
}

ITERATIONS_PER_MOVE = {
    "tictactoe": 500,
    "connect4": 1000,
}


def main(game_name: str):
    if game_name not in GAMES:
        print(f"Unknown game: {game_name}. Available: {', '.join(GAMES.keys())}")
        return

    GameClass = GAMES[game_name]
    iterations = ITERATIONS_PER_MOVE[game_name]

    state = GameClass.initial()
    move_number = 1

    print(f"\n{'='*50}")
    print(f"Playing {game_name.upper()} — MCTS vs MCTS")
    print(f"{'='*50}\n")

    while not state.is_terminal():
        print(f"--- Move {move_number} ({state.current_player} to play) ---")
        action = mcts_search(state, num_iterations=iterations)
        state = state.apply_action(action)
        print(f"Played: {action}")
        print(state.render())
        print()
        move_number += 1

    print(f"{'='*50}")
    winner = state._check_winner() if hasattr(state, "_check_winner") else state._winner() if hasattr(state, "_winner") else None
    if winner is not None:
        print(f"Winner: {winner}")
    else:
        print("Draw.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play MCTS vs MCTS")
    parser.add_argument("--game", default="tictactoe", help="Game to play: tictactoe or connect4")
    args = parser.parse_args()
    main(args.game)
