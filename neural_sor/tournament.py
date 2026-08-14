"""
Tournament Orchestrator for SOR AlphaZero.

Usage:
    # 1. Train all agents sequentially
    python tournament.py --train

    # 2. Run Round-Robin Tournament (assumes checkpoints exist)
    python tournament.py --eval --games 20 --sims 50
"""

import argparse
import subprocess
import os
import sys
from collections import defaultdict
from flax import nnx

from games.connect4 import Connect4
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from training.checkpoint import load_checkpoint, latest_checkpoint_index
from evaluate import play_one_game

OMEGAS = [1.0, 1.1, 1.25, 1.5, 1.7, 1.8]
INPUT_SIZE = 84
NUM_ACTIONS = 7
HIDDEN_SIZE = 128


def run_tournament(games_per_pair: int, sims: int):
    print(f"=== Running Round-Robin Tournament ===")
    print(f"Games per pair: {games_per_pair} | MCTS Sims: {sims}")
    
    # 1. Load all models
    models = {}
    for omega in OMEGAS:
        ckpt_dir = f"checkpoints_omega_{omega}"
        idx = latest_checkpoint_index(ckpt_dir)
        if idx is None:
            print(f"Warning: No checkpoint found for omega={omega} in {ckpt_dir}. Skipping.")
            continue
            
        model = AlphaZeroNet(
            input_size=INPUT_SIZE,
            num_actions=NUM_ACTIONS,
            hidden_size=HIDDEN_SIZE,
            omega=omega,
            rngs=nnx.Rngs(0)
        )
        load_checkpoint(model, ckpt_dir, idx)
        models[omega] = make_network_fn(model)
        
    if len(models) < 2:
        print("Not enough trained models to run a tournament.")
        return
        
    omegas_present = list(models.keys())
    
    # 2. Play matches
    # scores[omega] = total wins
    scores = defaultdict(int)
    # head_to_head[o1][o2] = wins of o1 against o2
    head_to_head = {o: defaultdict(int) for o in omegas_present}
    draws = 0
    
    # We want o1 and o2 to play `games_per_pair` times.
    # To be fair, they alternate being 'X' (first mover).
    
    for i in range(len(omegas_present)):
        for j in range(i + 1, len(omegas_present)):
            o1 = omegas_present[i]
            o2 = omegas_present[j]
            fn1 = models[o1]
            fn2 = models[o2]
            
            print(f"\n--- Matchup: Omega {o1} vs Omega {o2} ---")
            
            for g in range(games_per_pair):
                # Alternate first mover
                if g % 2 == 0:
                    # o1 is X, o2 is O
                    outcome = play_one_game(fn1, fn2, sims, sims) # sims_x, sims_o
                    if outcome == "X":
                        scores[o1] += 1
                        head_to_head[o1][o2] += 1
                    elif outcome == "O":
                        scores[o2] += 1
                        head_to_head[o2][o1] += 1
                    else:
                        draws += 1
                else:
                    # o2 is X, o1 is O
                    outcome = play_one_game(fn2, fn1, sims, sims)
                    if outcome == "X":
                        scores[o2] += 1
                        head_to_head[o2][o1] += 1
                    elif outcome == "O":
                        scores[o1] += 1
                        head_to_head[o1][o2] += 1
                    else:
                        draws += 1
                        
                print(f"  Game {g+1}/{games_per_pair} winner: {outcome} (X was {'O1' if g%2==0 else 'O2'})")

    # 3. Print Leaderboard
    print("\n" + "="*50)
    print("🏆 TOURNAMENT RESULTS 🏆")
    print("="*50)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (omega, wins) in enumerate(ranked):
        print(f"{rank+1}. Omega {omega:<4} - {wins} wins")
        
    print(f"\nTotal draws across tournament: {draws}")
    
    print("\n--- Head-to-Head Matrix (Row beats Column) ---")
    header = "      " + "".join([f"{o:^6}" for o in omegas_present])
    print(header)
    for o1 in omegas_present:
        row = f"{o1:<4} |"
        for o2 in omegas_present:
            if o1 == o2:
                row += "  --  "
            else:
                row += f"{head_to_head[o1][o2]:^6}"
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural-SOR Connect-4 Tournament")
    parser.add_argument("--train", action="store_true", help="Run sequential training for all omegas")
    parser.add_argument("--eval", action="store_true", help="Run round-robin evaluation tournament")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup in eval phase")
    parser.add_argument("--sims", type=int, default=50, help="MCTS sims per move in eval phase")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Target hardware device")
    args = parser.parse_args()

    OMEGAS = [1.0, 1.1, 1.25, 1.5, 1.7, 1.8]

    if args.train:
        print("=== Starting Sequential Training for all Omegas ===")
        for omega in OMEGAS:
            print(f"\n--- Training agent with omega={omega} ---")
            ckpt_dir = f"checkpoints_omega_{omega}"
            print(f"Checkpoints will be saved to: {ckpt_dir}")
            
            cmd = [
                sys.executable, "run_training.py",
                "--omega", str(omega),
                "--checkpoint-dir", ckpt_dir,
                "--device", args.device
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error training omega={omega}. Exiting.")
                sys.exit(1)
    elif args.eval:
        run_tournament(args.games, args.sims)
    else:
        print("Please specify --train or --eval.")
