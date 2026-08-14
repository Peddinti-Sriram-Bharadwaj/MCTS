"""
Interactive CLI game: Human vs AlphaZero AI.

Usage (from neural/ directory):
    python play.py                    # plays against latest checkpoint
    python play.py --checkpoint 200   # plays against ckpt_00200
    python play.py --sims 100         # set MCTS simulations for AI
"""

import argparse
from flax import nnx

from games.tictactoe import TicTacToe
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from mcts.search import mcts_search
from training.checkpoint import load_checkpoint, latest_checkpoint_index

INPUT_SIZE  = 18
NUM_ACTIONS = 9
HIDDEN_SIZE = 64

GRID_MAP = {
    0: "0 (top-left)", 1: "1 (top-mid)", 2: "2 (top-right)",
    3: "3 (mid-left)", 4: "4 (center)",  5: "5 (mid-right)",
    6: "6 (bot-left)", 7: "7 (bot-mid)", 8: "8 (bot-right)",
}


def print_board(state: TicTacToe):
    symbols = {"X": "X", "O": "O", None: "."}
    b = [symbols[x] for x in state.board]
    print("\n  Current Board:          Action Index Reference:")
    print(f"    {b[0]} | {b[1]} | {b[2]}                  0 | 1 | 2")
    print("   ---+---+---                ---+---+---")
    print(f"    {b[3]} | {b[4]} | {b[5]}                  3 | 4 | 5")
    print("   ---+---+---                ---+---+---")
    print(f"    {b[6]} | {b[7]} | {b[8]}                  6 | 7 | 8\n")


def human_turn(state: TicTacToe) -> int:
    legal = state.legal_actions()
    while True:
        try:
            val = input(f"Your turn ({state.current_player}). Enter move index {legal}: ")
            action = int(val.strip())
            if action in legal:
                return action
            print(f"Invalid move. Must be one of {legal}.")
        except (ValueError, KeyboardInterrupt):
            print("\nPlease enter a valid integer index.")


def ai_turn(state: TicTacToe, network_fn, num_simulations: int) -> int:
    print(f"AI ({state.current_player}) is thinking with {num_simulations} MCTS simulations...")
    action, pi, _ = mcts_search(state, network_fn, num_simulations, temperature=0.0)
    
    # Format policy distribution output
    top_moves = sorted(pi.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join([f"action {a}: {p*100:.1f}%" for a, p in top_moves if p > 0.01])
    print(f"  AI policy distribution: {top_str}")
    print(f"  AI chose move: {action}\n")
    return action


def main():
    parser = argparse.ArgumentParser(description="Play TicTacToe against trained AlphaZero model")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint index to load")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations for AI move")
    parser.add_argument("--human-first", action="store_true", help="Human plays first (as X)")
    args = parser.parse_args()

    ckpt_index = args.checkpoint
    if ckpt_index is None:
        ckpt_index = latest_checkpoint_index(args.checkpoint_dir)
        if ckpt_index is None:
            print(f"No checkpoints found in '{args.checkpoint_dir}'. Train first using run_training.py")
            return

    model = AlphaZeroNet(
        input_size=INPUT_SIZE, num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0)
    )
    load_checkpoint(model, args.checkpoint_dir, index=ckpt_index)
    network_fn = make_network_fn(model)

    human_player = "X" if args.human_first else "O"
    print(f"\n=== Playing TicTacToe vs AlphaZero (ckpt_{ckpt_index:05d}) ===")
    print(f"You are playing as: {human_player}\n")

    state = TicTacToe.initial()

    while not state.is_terminal():
        print_board(state)
        if state.current_player == human_player:
            action = human_turn(state)
        else:
            action = ai_turn(state, network_fn, args.sims)
        state = state.apply_action(action)

    print_board(state)
    winner = state._winner()
    if winner == human_player:
        print("🎉 Congratulations! You won!")
    elif winner is not None:
        print("🤖 AlphaZero AI won!")
    else:
        print("🤝 Game ended in a draw.")


if __name__ == "__main__":
    main()
