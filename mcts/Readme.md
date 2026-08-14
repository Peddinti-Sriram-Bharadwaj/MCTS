# Classical Monte Carlo Tree Search (MCTS)

This module provides a foundational, classical implementation of the Monte Carlo Tree Search algorithm. It relies entirely on random uniform rollouts for state evaluation, serving as an architectural baseline prior to the integration of neural network heuristics.

## Architectural Design

The implementation adheres to strict software engineering principles, primarily utilizing the **Strategy Pattern** (Dependency Injection) to completely decouple the search algorithm from the underlying game logic. 

### Game State Abstraction
- **`game_state_abstract.py`**: Defines the `GameState` Abstract Base Class (ABC). It enforces a strict contract dictating the methods any game must expose (`current_player`, `legal_actions`, `apply_action`, `is_terminal`, `reward`).
- **Concrete Implementations**: The `tictactoe.py` and `connect4.py` modules inherit from the `GameState` ABC. This allows the MCTS engine to seamlessly operate on either game environment with zero modifications to the core algorithm.

### MCTS Core Components
The search algorithm is modularized into distinct theoretical phases:
- **`node.py`**: Defines the fundamental tree structure, tracking visit counts and aggregate state values.
- **`selection.py`**: Implements the Upper Confidence Bound applied to Trees (UCT/UCB1) formula to traverse the tree, balancing exploration and exploitation.
- **`expansion.py`**: Instantiates new child nodes when the selection phase reaches the frontier.
- **`simulation.py`**: Executes a uniform random rollout from the expanded node to a terminal state to sample the true game outcome.
- **`backprop.py`**: Propagates the sampled terminal reward back up the search path, mathematically alternating the perspective (sign inversion) at each depth level to account for zero-sum alternating turns.
- **`search.py`**: The central orchestrator that manages the four-phase MCTS loop.

## Execution Guide

The `play.py` script serves as the primary entry point for executing self-play games. The target game environment is injected at runtime via command-line arguments.

### Dependencies
This module strictly utilizes the Python Standard Library. No external packages are required.

### Self-Play Demonstrations

Execute Tic-Tac-Toe with 500 simulations per move:
```bash
python3 play.py --game tictactoe
```

Execute Connect-4 with 1000 simulations per move:
```bash
python3 play.py --game connect4
```

## Extending the Framework

To introduce a novel game environment into the search engine:
1. Construct a class that strictly inherits from and implements the `GameState` abstract interface.
2. Register the new class within the `GAMES` dictionary located in the `play.py` driver.
3. Execute the driver with the newly assigned string key.

Due to the Dependency Injection architecture, the core MCTS algorithm requires zero modification to solve the new environment.
