import argparse
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from mcts.search import mcts_search
from training.self_play import self_play_game
from training.train import train_step

def get_game_class(game_name):
    if game_name == "connect4":
        return Connect4, 84, 7
    elif game_name == "tictactoe":
        return TicTacToe, 18, 9
    else:
        raise ValueError("Unknown game")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="connect4")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    GameClass, input_size, num_actions = get_game_class(args.game)
    hidden_size = 128 if args.game == "connect4" else 64

    # Force JAX initialization to lock in the backend
    backend = jax.default_backend().upper()
    device = str(jax.devices()[0])
    
    metrics = {
        "hardware": {
            "backend": backend,
            "device": device
        },
        "forward_pass": {},
        "mcts": 0.0,
        "self_play": 0.0,
        "training": 0.0
    }

    model = AlphaZeroNet(input_size, num_actions, hidden_size, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)

    print(f"[{backend}] Profiling Forward Pass...")
    batched_forward = nnx.vmap(model, in_axes=0, out_axes=0)
    
    # Needs explicit JIT to measure accurately
    @nnx.jit
    def fast_forward(m, x):
        return nnx.vmap(m, in_axes=0, out_axes=0)(x)

    # Compile with dummy data
    dummy_x = jnp.zeros((1, input_size))
    _ = fast_forward(model, dummy_x)

    batch_sizes = [1, 8, 64, 256]
    for b in batch_sizes:
        x = jnp.zeros((b, input_size))
        # Warmup specific shape
        _ = fast_forward(model, x)
        
        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            logits, val = fast_forward(model, x)
            # block until execution finishes
            logits.block_until_ready()
        end = time.perf_counter()
        
        metrics["forward_pass"][str(b)] = ((end - start) / iters) * 1000 # ms per batch

    print(f"[{backend}] Profiling MCTS...")
    state = GameClass.initial()
    
    # Warmup
    mcts_search(state, network_fn, 5)
    
    start = time.perf_counter()
    mcts_search(state, network_fn, 50)
    end = time.perf_counter()
    metrics["mcts"] = (end - start) * 1000 # ms for 50 sims

    print(f"[{backend}] Profiling Self-Play Game...")
    start = time.perf_counter()
    self_play_game(state, network_fn, num_simulations=20)
    end = time.perf_counter()
    metrics["self_play"] = (end - start) * 1000 # ms for full game

    print(f"[{backend}] Profiling Training Step...")
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Dummy batch
    states = np.zeros((64, input_size), dtype=np.float32)
    pi_vecs = np.ones((64, num_actions), dtype=np.float32) / num_actions
    zs = np.zeros((64,), dtype=np.float32)
    
    # Warmup
    train_step(model, optimizer, states[:2], pi_vecs[:2], zs[:2])
    
    start = time.perf_counter()
    iters = 10
    for _ in range(iters):
        train_step(model, optimizer, states, pi_vecs, zs)
    end = time.perf_counter()
    
    metrics["training"] = ((end - start) / iters) * 1000 # ms per train step

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[{backend}] Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
