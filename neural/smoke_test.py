"""
Smoke test — run from neural/ directory:
    python smoke_test.py

Checks:
  1. TicTacToe encodes correctly to (2,3,3)
  2. AlphaZeroNet forward pass produces correct output shapes
  3. make_network_fn returns valid (policy_probs, value)
  4. expand() populates all children with priors
  5. mcts_search() runs and returns (action, pi) without error
  6. self_play_game() completes a full game and returns examples
  7. ReplayBuffer stores and samples correctly
"""

import sys
import traceback
import jax
import jax.numpy as jnp
from flax import nnx

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except Exception:
        print(f"{FAIL}  {name}")
        traceback.print_exc()
        return False


# ── 1. TicTacToe encoding ────────────────────────────────────────────────────

def test_encoding():
    from games.tictactoe import TicTacToe
    state = TicTacToe.initial()
    arr = state.to_array()
    assert arr.shape == (2, 3, 3), f"Expected (2,3,3), got {arr.shape}"
    assert arr.dtype == jnp.float32
    # Empty board: all zeros
    assert jnp.all(arr == 0), "Empty board should encode as all zeros"
    # After one move by X
    state2 = state.apply_action(4)  # centre
    arr2 = state2.to_array()
    # Now it's O's turn — plane 1 (opponent=X) should have a 1 at centre
    assert arr2[1].flatten()[4] == 1.0, "X's piece should appear in opponent plane after X moves"


# ── 2. Network forward pass ───────────────────────────────────────────────────

def test_network_shapes():
    from network.model import AlphaZeroNet
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    x = jnp.zeros((18,))
    logits, value = model(x)
    assert logits.shape == (9,), f"Expected (9,), got {logits.shape}"
    assert value.shape == (1,), f"Expected (1,), got {value.shape}"
    assert -1.0 <= float(value[0]) <= 1.0, "Value should be in [-1, 1] (tanh output)"


# ── 3. make_network_fn ────────────────────────────────────────────────────────

def test_network_fn():
    from games.tictactoe import TicTacToe
    from network.model import AlphaZeroNet
    from network.inference import make_network_fn
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = TicTacToe.initial()
    probs, value = network_fn(state)
    assert probs.shape == (9,), f"Expected (9,), got {probs.shape}"
    assert abs(float(probs.sum()) - 1.0) < 1e-5, f"Probs should sum to 1, got {probs.sum()}"
    assert -1.0 <= value <= 1.0


# ── 4. expand() ───────────────────────────────────────────────────────────────

def test_expand():
    from games.tictactoe import TicTacToe
    from network.model import AlphaZeroNet
    from network.inference import make_network_fn
    from mcts.node import Node
    from mcts.expansion import expand
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = TicTacToe.initial()
    root = Node(state=state)
    value = expand(root, network_fn)
    assert len(root.children) == 9, f"Empty board should have 9 children, got {len(root.children)}"
    prior_sum = sum(c.prior_prob for c in root.children.values())
    assert abs(prior_sum - 1.0) < 1e-5, f"Priors should sum to 1, got {prior_sum}"
    assert isinstance(value, float)


# ── 5. mcts_search() ─────────────────────────────────────────────────────────

def test_mcts_search():
    from games.tictactoe import TicTacToe
    from network.model import AlphaZeroNet
    from network.inference import make_network_fn
    from mcts.search import mcts_search
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = TicTacToe.initial()
    action, pi = mcts_search(state, network_fn, num_simulations=10)
    assert action in range(9), f"Action {action} out of range"
    assert abs(sum(pi.values()) - 1.0) < 1e-5, f"pi should sum to 1, got {sum(pi.values())}"


# ── 6. self_play_game() ───────────────────────────────────────────────────────

def test_self_play():
    from games.tictactoe import TicTacToe
    from network.model import AlphaZeroNet
    from network.inference import make_network_fn
    from training.self_play import self_play_game
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    examples = self_play_game(TicTacToe.initial(), network_fn, num_simulations=5)
    assert len(examples) >= 1, "Game should produce at least 1 example"
    assert len(examples) <= 9, "TicTacToe can't exceed 9 moves"
    state_arr, pi, z = examples[0]
    assert state_arr.shape == (18,), f"Expected flat (18,), got {state_arr.shape}"
    assert z in (-1.0, 0.0, 1.0), f"Unexpected z value: {z}"


# ── 7. ReplayBuffer ───────────────────────────────────────────────────────────

def test_replay_buffer():
    from games.tictactoe import TicTacToe
    from network.model import AlphaZeroNet
    from network.inference import make_network_fn
    from training.self_play import self_play_game
    from training.replay_buffer import ReplayBuffer
    model = AlphaZeroNet(input_size=18, num_actions=9, hidden_size=32, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    examples = self_play_game(TicTacToe.initial(), network_fn, num_simulations=5)
    buf = ReplayBuffer(capacity=1000, num_actions=9)
    buf.add_game(examples)
    assert len(buf) == len(examples)
    states, pi_vecs, zs = buf.sample(batch_size=len(examples))
    assert states.shape == (len(examples), 18)
    assert pi_vecs.shape == (len(examples), 9)
    assert zs.shape == (len(examples),)


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Neural MCTS Smoke Test ===\n")
    tests = [
        ("TicTacToe encoding",    test_encoding),
        ("Network forward pass",  test_network_shapes),
        ("make_network_fn",       test_network_fn),
        ("expand() — all children populated", test_expand),
        ("mcts_search() — 10 simulations",    test_mcts_search),
        ("self_play_game() — full game",       test_self_play),
        ("ReplayBuffer — store + sample",      test_replay_buffer),
    ]

    results = [check(name, fn) for name, fn in tests]
    passed = sum(results)
    total  = len(results)

    print(f"\n{passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)
