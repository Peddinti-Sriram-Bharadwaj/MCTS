"""
Run AlphaZero-style training on TicTacToe.

Usage (from neural/ directory):
    python run_training.py
"""

from flax import nnx
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from games.connect4 import Connect4
from training.train import train_loop

# ── Hyperparameters ───────────────────────────────────────────────────────────

INPUT_SIZE          = 84    # 2 planes * 6*7 board, flattened
NUM_ACTIONS         = 7     # Connect-4 action space
HIDDEN_SIZE         = 128   # shared trunk width

NUM_ITERATIONS      = 200   # self-play + train cycles
GAMES_PER_ITERATION = 20    # self-play games per cycle
NUM_SIMULATIONS     = 100   # MCTS simulations per move
BATCH_SIZE          = 64    # minibatch size
BUFFER_CAPACITY     = 20_000
LEARNING_RATE       = 1e-3

# ── Hardware Info ─────────────────────────────────────────────────────────────
import jax
print(f"\n[Hardware Info]")
print(f"  Backend: {jax.default_backend().upper()}")
print(f"  Devices: {jax.devices()}")
print(f"  Local Devices: {jax.local_device_count()} | Total Devices: {jax.device_count()}")
print(f"  JAX is utilizing: {'Apple Silicon (GPU)' if jax.default_backend() in ['metal', 'mps'] else 'CPU only (Unaccelerated)'}")
print(f"  Parallelization: nnx.vmap across {BATCH_SIZE} elements per batch")
print("────────────────────────────────────────────────────────────────────────\n")

# ── Build model ───────────────────────────────────────────────────────────────

model = AlphaZeroNet(
    input_size  = INPUT_SIZE,
    num_actions = NUM_ACTIONS,
    hidden_size = HIDDEN_SIZE,
    rngs        = nnx.Rngs(0),   # seed=0 for reproducibility
)

print("Model built.")
print(f"  Trunk:       Linear({INPUT_SIZE} -> {HIDDEN_SIZE}) x2")
print(f"  Policy head: Linear({HIDDEN_SIZE} -> {NUM_ACTIONS})")
print(f"  Value head:  Linear({HIDDEN_SIZE} -> 1) + tanh")
print(f"\nTraining config:")
print(f"  {NUM_ITERATIONS} iterations × {GAMES_PER_ITERATION} games")
print(f"  {NUM_SIMULATIONS} MCTS simulations per move")
print(f"  Batch size: {BATCH_SIZE}  |  Buffer: {BUFFER_CAPACITY}")
print(f"  Learning rate: {LEARNING_RATE}\n")
print("=" * 50)

# ── Train ─────────────────────────────────────────────────────────────────────

train_loop(
    model               = model,
    initial_state_fn    = Connect4.initial,
    num_iterations      = NUM_ITERATIONS,
    games_per_iteration = GAMES_PER_ITERATION,
    num_simulations     = NUM_SIMULATIONS,
    batch_size          = BATCH_SIZE,
    buffer_capacity     = BUFFER_CAPACITY,
    learning_rate       = LEARNING_RATE,
)

print("=" * 50)
print("Training complete.")
