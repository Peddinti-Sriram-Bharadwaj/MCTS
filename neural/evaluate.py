"""
Evaluate a checkpointed model against a random-weight baseline.

Usage (from neural/ directory):
    python evaluate.py                    # loads latest checkpoint
    python evaluate.py --checkpoint 50    # loads ckpt_00050
    python evaluate.py --checkpoint 50 --games 50 --sims 100
    python evaluate.py --list             # list available checkpoints

The trained model always plays both sides (X and O) across game pairs
to cancel out first-mover advantage. Results are reported per side.
"""

import argparse
from flax import nnx

from network.model import AlphaZeroNet
from network.inference import make_network_fn
from games.tictactoe import TicTacToe
from mcts.search import mcts_search
from training.checkpoint import load_checkpoint, list_checkpoints, latest_checkpoint_index

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
INPUT_SIZE     = 18
NUM_ACTIONS    = 9
HIDDEN_SIZE    = 64

# ── Game runner ───────────────────────────────────────────────────────────────

def play_one_game(fn_x, fn_o, sims_x: int, sims_o: int) -> str:
    """Play one TicTacToe game. Returns 'X', 'O', or 'draw'."""
    state = TicTacToe.initial()
    fns  = {"X": fn_x,   "O": fn_o}
    sims = {"X": sims_x, "O": sims_o}
    while not state.is_terminal():
        action, _ = mcts_search(state, fns[state.current_player], sims[state.current_player])
        state = state.apply_action(action)
    winner = state._winner()
    return winner if winner is not None else "draw"


def run_evaluation(
    trained_fn, random_fn,
    num_games: int,
    sims_trained: int,
    sims_random: int,
):
    """Play num_games games, alternating who goes first each pair."""
    results = {"trained": 0, "random": 0, "draw": 0}

    for i in range(num_games):
        # Alternate: trained=X on even games, trained=O on odd games
        if i % 2 == 0:
            outcome = play_one_game(trained_fn, random_fn, sims_trained, sims_random)
            if outcome == "X":     results["trained"] += 1
            elif outcome == "O":   results["random"]  += 1
            else:                  results["draw"]    += 1
        else:
            outcome = play_one_game(random_fn, trained_fn, sims_random, sims_trained)
            if outcome == "O":     results["trained"] += 1
            elif outcome == "X":   results["random"]  += 1
            else:                  results["draw"]    += 1

        side = "X" if i % 2 == 0 else "O"
        t, r, d = results["trained"], results["random"], results["draw"]
        print(f"  game {i+1:3d} (trained={side}): {outcome:4s}  [T={t} R={r} D={d}]")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero checkpoint vs random baseline")
    parser.add_argument(
        "--checkpoint", type=int, default=None,
        help="Checkpoint index to load (default: latest)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {CHECKPOINT_DIR})"
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Number of games to play (default: 20)"
    )
    parser.add_argument(
        "--sims", type=int, default=50,
        help="MCTS simulations per move for both players (default: 50)"
    )
    parser.add_argument(
        "--sims-trained", type=int, default=None,
        help="Override sims for trained model only (useful for low-sim tests)"
    )
    parser.add_argument(
        "--sims-random", type=int, default=None,
        help="Override sims for random baseline only"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available checkpoints and exit"
    )
    args = parser.parse_args()

    # -- List mode
    if args.list:
        indices = list_checkpoints(args.checkpoint_dir)
        if not indices:
            print(f"No checkpoints found in '{args.checkpoint_dir}'")
        else:
            latest = max(indices)
            print(f"Available checkpoints in '{args.checkpoint_dir}':")
            for idx in indices:
                marker = " ← latest" if idx == latest else ""
                print(f"  ckpt_{idx:05d}{marker}")
        return

    # -- Resolve checkpoint index
    ckpt_index = args.checkpoint
    if ckpt_index is None:
        ckpt_index = latest_checkpoint_index(args.checkpoint_dir)
        if ckpt_index is None:
            print(f"No checkpoints found in '{args.checkpoint_dir}'. Train first.")
            return
        print(f"No --checkpoint specified. Using latest: {ckpt_index}")

    # -- Load trained model
    trained_model = AlphaZeroNet(
        input_size=INPUT_SIZE, num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0)
    )
    load_checkpoint(trained_model, args.checkpoint_dir, index=ckpt_index)

    # -- Build random baseline (different seed → different weights)
    random_model = AlphaZeroNet(
        input_size=INPUT_SIZE, num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(99)
    )

    # -- Resolve per-player sim counts
    sims_trained = args.sims_trained if args.sims_trained is not None else args.sims
    sims_random  = args.sims_random  if args.sims_random  is not None else args.sims

    trained_fn = make_network_fn(trained_model)
    random_fn  = make_network_fn(random_model)

    # -- Run evaluation
    print(f"\n{'='*55}")
    print(f"Evaluating checkpoint {ckpt_index} vs random baseline")
    print(f"  games={args.games}  sims_trained={sims_trained}  sims_random={sims_random}")
    print(f"{'='*55}\n")

    results = run_evaluation(trained_fn, random_fn, args.games, sims_trained, sims_random)

    total = sum(results.values())
    print(f"\n{'='*55}")
    print(f"Results over {total} games:")
    print(f"  Trained wins : {results['trained']:3d}  ({100*results['trained']/total:.1f}%)")
    print(f"  Random  wins : {results['random']:3d}  ({100*results['random']/total:.1f}%)")
    print(f"  Draws        : {results['draw']:3d}  ({100*results['draw']/total:.1f}%)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
