import argparse
import os
import sys

# ── Parse Args BEFORE importing JAX ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train AlphaZero with SOR")
parser.add_argument("--omega", type=float, default=1.25, help="SOR relaxation parameter")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Target hardware device")
# Parse args early. Since we might be called by tournament.py or directly, we parse known args.
args, _ = parser.parse_known_args()

if args.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    # Remove it if it was set to cpu by a parent script, letting JAX find GPU
    if "JAX_PLATFORMS" in os.environ:
        del os.environ["JAX_PLATFORMS"]

# Now we can safely import jax and flax
import jax
from flax import nnx
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from games.connect4 import Connect4
from training.train import train_loop

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_SIZE          = 84    # 2 planes * 6*7 board, flattened
NUM_ACTIONS         = 7     # Connect-4 action space
HIDDEN_SIZE         = 128   # increased shared trunk width for more complex game

NUM_ITERATIONS      = 200   # self-play + train cycles
GAMES_PER_ITERATION = 20    # self-play games per cycle
NUM_SIMULATIONS     = 100   # MCTS simulations per move
BATCH_SIZE          = 64    # minibatch size
BUFFER_CAPACITY     = 20_000
LEARNING_RATE       = 1e-3

# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Neural SOR MCTS Training (Connect-4, omega={args.omega}) ===")
    
    print(f"\n[Hardware Info]")
    print(f"  Backend: {jax.default_backend().upper()}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Local Devices: {jax.local_device_count()} | Total Devices: {jax.device_count()}")
    print(f"  JAX is utilizing: {'Apple Silicon (GPU)' if jax.default_backend() in ['metal', 'mps'] else 'CPU only (Unaccelerated)'}")
    print(f"  Parallelization: nnx.vmap across {BATCH_SIZE} elements per batch")
    print("────────────────────────────────────────────────────────────────────────\n")
    
    # 1. Initialize neural network
    model = AlphaZeroNet(
        input_size=INPUT_SIZE,
        num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN_SIZE,
        omega=args.omega,
        rngs=nnx.Rngs(0)
    )

    print(f"Model Architecture:")
    print(f"  Shared trunk: Linear({INPUT_SIZE} -> {HIDDEN_SIZE}) -> Linear({HIDDEN_SIZE} -> {HIDDEN_SIZE})")
    print(f"  Policy head: Linear({HIDDEN_SIZE} -> {NUM_ACTIONS})")
    print(f"  Value head (SOR): Linear({HIDDEN_SIZE} -> 1) x 2 + tanh")
    print(f"\nTraining config:")
    print(f"  {NUM_ITERATIONS} iterations × {GAMES_PER_ITERATION} games")
    print(f"  {NUM_SIMULATIONS} MCTS simulations per move")
    print(f"  Batch size: {BATCH_SIZE}  |  Buffer: {BUFFER_CAPACITY}")
    print(f"  Learning rate: {LEARNING_RATE}\n")

    # 2. Launch training loop
    train_loop(
        model=model,
        initial_state_fn=Connect4.initial,
        num_iterations=NUM_ITERATIONS,
        games_per_iteration=GAMES_PER_ITERATION,
        num_simulations=NUM_SIMULATIONS,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=10,
        omega_target=args.omega
    )

if __name__ == "__main__":
    main()
