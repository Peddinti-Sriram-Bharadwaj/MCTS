"""
Run AlphaZero-style training on TicTacToe.

Usage (from neural/ directory):
    python run_training.py
"""

from flax import nnx
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from games.tictactoe import TicTacToe
from training.train import train_loop

# ── Hyperparameters ───────────────────────────────────────────────────────────

INPUT_SIZE          = 18    # 2 planes * 3*3 board, flattened
NUM_ACTIONS         = 9     # TicTacToe action space
HIDDEN_SIZE         = 64    # shared trunk width

NUM_ITERATIONS      = 100   # self-play + train cycles
GAMES_PER_ITERATION = 5     # self-play games per cycle
NUM_SIMULATIONS     = 50    # MCTS simulations per move
BATCH_SIZE          = 64    # minibatch size
BUFFER_CAPACITY     = 10_000
LEARNING_RATE       = 1e-3

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
print(f"  {NUM_ITERATIONS} iterations × {GAMES_PER_ITERATION} games × up to 9 moves")
print(f"  {NUM_SIMULATIONS} MCTS simulations per move")
print(f"  Batch size: {BATCH_SIZE}  |  Buffer: {BUFFER_CAPACITY}")
print(f"  Learning rate: {LEARNING_RATE}\n")
print("=" * 50)

# ── Train ─────────────────────────────────────────────────────────────────────

train_loop(
    model               = model,
    initial_state_fn    = TicTacToe.initial,
    num_iterations      = NUM_ITERATIONS,
    games_per_iteration = GAMES_PER_ITERATION,
    num_simulations     = NUM_SIMULATIONS,
    batch_size          = BATCH_SIZE,
    buffer_capacity     = BUFFER_CAPACITY,
    learning_rate       = LEARNING_RATE,
)

print("=" * 50)
print("Training complete.")
