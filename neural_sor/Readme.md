# AlphaZero with Successive Over-Relaxation (SOR) in JAX and Flax

This repository provides an extended implementation of the AlphaZero algorithm, incorporating Successive Over-Relaxation (SOR) into the value head's loss function to accelerate convergence and stabilize Q-value targets. The implementation utilizes Monte Carlo Tree Search (MCTS) with a dual-head neural network via JAX and Flax NNX.

## Project Architecture

```text
neural-SOR/
├── games/
│   ├── base.py          # Abstract GameState interface 
│   ├── connect4.py      # Connect-4 implementation
│   └── tictactoe.py     # Tic-Tac-Toe implementation
├── mcts/
│   ├── node.py          # MCTS Search tree node definition
│   ├── selection.py     # PUCT search selection logic 
│   ├── expansion.py     # Neural network expansion and prior probability initialization
│   ├── backpropagation.py # Value propagation up the search tree
│   └── search.py        # MCTS search algorithm implementation
├── network/
│   ├── model.py         # Flax NNX AlphaZeroNet with split Value Heads for SOR
│   └── inference.py     # JIT-compiled network forward pass wrapper
├── training/
│   ├── self_play.py     # Self-play trajectory generation
│   ├── replay_buffer.py # Transition ring buffer
│   ├── train.py         # Optax Adam optimizer and SOR-modified loss function
│   └── checkpoint.py    # Orbax checkpoint manager
├── run_training.py      # Entry point for single-agent self-play training
├── tournament.py        # Orchestrator for hyperparameter sweeping and evaluation
├── evaluate.py          # Head-to-head evaluation suite
├── play.py              # Interactive terminal interface
├── smoke_test.py        # Component verification test suite
└── requirements.txt     # Dependency specifications
```

## Technical Specifications

### Successive Over-Relaxation (SOR) Integration
This repository diverges from standard AlphaZero by applying Successive Over-Relaxation (SOR) strictly to the Value Head. This is controlled by the relaxation parameter $\omega$. 
- When $\omega = 1.0$, the network operates as a standard AlphaZero implementation.
- When $\omega > 1.0$, the network applies Over-Relaxation to extrapolate gradient updates.
- The Value Head computes `V(s) = fc2(fc1(s))`. The SOR transformation is applied between `fc1` and `fc2` to stabilize the intermediate representations.

### Base Architecture
1. **JAX and Flax NNX Integration**: The architecture utilizes the Flax NNX object-oriented functional module paradigm.
2. **Dual-Head Architecture**: A unified backbone model branch computes two distinct outputs:
   - **Policy Head**: Outputs move logit distributions over the action space.
   - **Value Head (SOR)**: Predicts the expected state evaluation in the continuous interval `[-1, 1]` using a `tanh` activation function, modified by $\omega$.
3. **MCTS Configuration**: Utilizes PUCT Selection, Dirichlet Noise exploration, and Temperature Sampling.
4. **Checkpointing**: Orbax is utilized for saving and restoring model parameters systematically.

## Usage Guide

### 1. Installation

Activate the Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Verification

Execute the component verification suite to ensure hardware and software integrity:

```bash
python smoke_test.py
```

### 3. Hyperparameter Sweeping (Tournament Mode)

The primary entry point for this repository is the `tournament.py` orchestrator, which sequentially trains and evaluates multiple agents across varying $\omega$ values.

**Phase 1: Sequential Training**
```bash
python tournament.py --train --device cpu
```
This script sequentially trains independent agents for $\omega \in [1.0, 1.1, 1.25, 1.5, 1.7, 1.8]$ and saves them to isolated checkpoint directories.

**Phase 2: Round-Robin Evaluation**
```bash
python tournament.py --eval --games 20 --sims 50
```
This script pits every fully trained agent against every other agent in an alternating-first-mover format to empirically determine the optimal relaxation parameter. It produces a comprehensive head-to-head win matrix.

### 4. Single-Agent Training

To train a single agent manually with a specific $\omega$ parameter:

```bash
python run_training.py --omega 1.5 --checkpoint-dir checkpoints_omega_1.5 --device cpu
```

### 5. Interactive Execution

Play against the trained model in an interactive terminal session:

```bash
python play.py                           
python play.py --checkpoint 190 --sims 50 
python play.py --human-first              
```

## Game Interface Abstraction

To implement additional games, inherit from the base `GameState` interface defined in `games/base.py`:

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
