# AlphaZero Neural MCTS in JAX / Flax NNX

A modular, clean implementation of AlphaZero combining **Monte Carlo Tree Search (MCTS)** with a **dual-head policy & value neural network** using **JAX and Flax NNX**.

---

## 🏛️ Project Architecture

```
neural/
├── games/
│   ├── base.py          # Abstract GameState interface (to_array, legal_actions, reward)
│   └── tictactoe.py     # Canonical TicTacToe implementation & tensor board encoding
├── mcts/
│   ├── node.py          # MCTS Search tree node
│   ├── selection.py     # PUCT search selection logic (with zero-visit prior preservation)
│   ├── expansion.py     # Neural network expansion & prior probability initialization
│   ├── backpropagation.py # Value propagation up search tree
│   └── search.py        # MCTS search algorithm (Dirichlet noise & temperature sampling)
├── network/
│   ├── model.py         # Flax NNX AlphaZeroNet (shared trunk, policy & value heads)
│   └── inference.py     # @nnx.jit-accelerated network forward pass wrapper
├── training/
│   ├── self_play.py     # Self-play trajectory collector
│   ├── replay_buffer.py # Fixed-capacity transition ring buffer
│   ├── train.py         # Optax Adam optimizer & AlphaZero training step
│   └── checkpoint.py    # Orbax checkpoint manager (indexed save/load)
├── run_training.py      # Entry point for self-play training
├── evaluate.py          # Head-to-head evaluation suite (vs random baseline)
├── play.py              # Interactive Human vs AI terminal interface
├── smoke_test.py        # Verification test suite
└── requirements.txt     # Dependency specifications
```

---

## ⚡ Key Architectural Features

1. **JAX + Flax NNX**: Utilizes Flax NNX object-oriented functional module paradigm (`nnx.Module`, `@nnx.jit`, `nnx.vmap`, `nnx.Optimizer`).
2. **Shared Trunk Dual-Head Network**: A unified backbone model branching into:
   - **Policy Head**: Outputs move logit distributions over `num_actions`.
   - **Value Head**: Predicts expected position evaluation in `[-1, 1]` with `tanh` activation.
3. **AlphaZero MCTS**:
   - **PUCT Selection**: Combines state action value $Q$ with policy prior bonus $U(s, a)$.
   - **Dirichlet Exploration Noise**: Injected into root node priors during training (`dirichlet_alpha=1.11`, `epsilon=0.25`).
   - **Temperature Sampling**: Move selection during training samples from visit distributions $p_i \propto N(s, a)^{1/\tau}$.
4. **Orbax Checkpointing**: Save and restore model weights with versioned indices (`checkpoints/ckpt_00100`).

---

## 🚀 Quick Start & Usage

### 1. Installation

Activate your environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Verification (Smoke Test)

Run the end-to-end component verification suite:

```bash
python smoke_test.py
```

### 3. Training

Launch the self-play training loop:

```bash
python run_training.py
```
Checkpoints will automatically be saved to `checkpoints/ckpt_XXXXX`.

### 4. Evaluation

Evaluate a trained checkpoint against a random baseline:

```bash
# List available checkpoints
python evaluate.py --list

# Evaluate pure policy network performance (0 simulations)
python evaluate.py --sims 0 --games 40

# Evaluate MCTS-assisted performance
python evaluate.py --checkpoint 190 --sims 50 --games 20
```

### 5. Interactive Play vs AI

Play TicTacToe in your terminal against the trained model:

```bash
python play.py                           # plays against latest checkpoint
python play.py --checkpoint 190 --sims 50 # specify checkpoint index and MCTS depth
python play.py --human-first              # human plays first as 'X'
```

---

## 🎮 Game Interface Abstraction (`games/base.py`)

To implement a new game (e.g. Connect-4 or Chess), inherit from `GameState`:

```python
class GameState(ABC):
    @property
    @abstractmethod
    def current_player(self) -> str: ...
    @abstractmethod
    def legal_actions(self) -> list[int]: ...
    @abstractmethod
    def apply_action(self, action: int) -> "GameState": ...
    @abstractmethod
    def is_terminal(self) -> bool: ...
    @abstractmethod
    def reward(self) -> float: ...
    @property
    @abstractmethod
    def num_actions(self) -> int: ...
    @abstractmethod
    def to_array(self) -> jnp.ndarray: ...
```
