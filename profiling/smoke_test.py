from games.connect4 import Connect4
from network.model import AlphaZeroNet
from network.inference import make_network_fn
from mcts.node import Node
from mcts.expansion import expand
from mcts.search import mcts_search
from training.replay_buffer import ReplayBuffer
from training.self_play import self_play_game
import jax.numpy as jnp
from flax import nnx
import traceback

# Configuration for Connect4
INPUT_SIZE = 84
NUM_ACTIONS = 7
HIDDEN_SIZE = 64

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return False

def test_encoding():
    state = Connect4.initial()
    arr = state.to_array()
    assert arr.shape == (2, 6, 7), f"Expected (2, 6, 7) got {arr.shape}"
    assert arr.dtype == jnp.float32, f"Expected float32 got {arr.dtype}"

def test_network():
    model = AlphaZeroNet(
        input_size=INPUT_SIZE, num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0)
    )
    dummy_input = jnp.zeros((1, INPUT_SIZE))
    logits, value = model(dummy_input)
    assert logits.shape == (1, NUM_ACTIONS)
    assert value.shape == (1, 1)

def test_inference_fn():
    model = AlphaZeroNet(input_size=INPUT_SIZE, num_actions=NUM_ACTIONS, hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = Connect4.initial()
    pi, val = network_fn(state)
    assert len(pi) == NUM_ACTIONS
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0

def test_expand():
    model = AlphaZeroNet(input_size=INPUT_SIZE, num_actions=NUM_ACTIONS, hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = Connect4.initial()
    root = Node(state=state)
    value = expand(root, network_fn)
    assert len(root.children) == NUM_ACTIONS, f"Empty board should have {NUM_ACTIONS} children, got {len(root.children)}"
    prior_sum = sum(c.prior_prob for c in root.children.values())
    assert abs(prior_sum - 1.0) < 1e-5, f"Priors should sum to 1, got {prior_sum}"
    assert isinstance(value, float)

def test_mcts_search():
    model = AlphaZeroNet(input_size=INPUT_SIZE, num_actions=NUM_ACTIONS, hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = Connect4.initial()
    action, pi = mcts_search(state, network_fn, num_simulations=10)
    assert action in range(NUM_ACTIONS)
    assert sum(pi.values()) > 0

def test_self_play():
    model = AlphaZeroNet(input_size=INPUT_SIZE, num_actions=NUM_ACTIONS, hidden_size=HIDDEN_SIZE, rngs=nnx.Rngs(0))
    network_fn = make_network_fn(model)
    state = Connect4.initial()
    examples = self_play_game(state, network_fn, num_simulations=5)
    assert len(examples) > 0
    state_arr, pi, z = examples[0]
    assert state_arr.shape == (INPUT_SIZE,)
    assert isinstance(pi, dict)
    assert isinstance(z, float)

def test_buffer():
    buffer = ReplayBuffer(capacity=100, num_actions=NUM_ACTIONS)
    import numpy as np
    examples = [
        (np.zeros(INPUT_SIZE), {0: 0.5, 1: 0.5}, 1.0),
        (np.zeros(INPUT_SIZE), {2: 1.0}, -1.0)
    ]
    buffer.add_game(examples)
    assert len(buffer) == 2
    s, p, z = buffer.sample(2)
    assert s.shape == (2, INPUT_SIZE)
    assert p.shape == (2, NUM_ACTIONS)
    assert z.shape == (2,)

def test_hardware():
    import jax
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"\n  [Hardware] Backend: {backend.upper()}")
    print(f"  [Hardware] Devices: {', '.join(str(d) for d in devices)}")
    print(f"  [Hardware] Local devices: {jax.local_device_count()} | Total: {jax.device_count()}")
    # Optionally assert that metal is being used, but we won't strictly fail if it's cpu.
    
def run_smoke_test():
    print("\n=== Neural MCTS Smoke Test (Connect4) ===\n")
    tests = [
        ("Hardware verification", test_hardware),
        ("Connect4 encoding", test_encoding),
        ("Network forward pass", test_network),
        ("make_network_fn", test_inference_fn),
        ("expand() — all children populated", test_expand),
        ("mcts_search() — 10 simulations", test_mcts_search),
        ("self_play_game() — full game", test_self_play),
        ("ReplayBuffer — store + sample", test_buffer),
    ]
    
    passed = 0
    for name, fn in tests:
        if check(name, fn):
            passed += 1
            
    print(f"\n{passed}/{len(tests)} tests passed.\n")

if __name__ == "__main__":
    run_smoke_test()
