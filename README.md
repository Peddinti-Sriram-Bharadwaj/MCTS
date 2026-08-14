# Monte Carlo Tree Search (MCTS) Algorithm Suite

This repository contains a progression of Monte Carlo Tree Search implementations for two-player, zero-sum board games, ranging from foundational heuristic-driven architectures to state-of-the-art Deep Reinforcement Learning models (AlphaZero).

## Repository Architecture

The repository is modularized into distinct directories, each demonstrating an evolution in algorithmic complexity:

- **`mcts/`**: A modular, heuristic-driven Monte Carlo Tree Search engine implemented via the Strategy Pattern (Dependency Injection). This serves as the foundational classical MCTS algorithm capable of playing Connect-4 and Tic-Tac-Toe.
- **`neural/`**: A Deep Reinforcement Learning implementation of AlphaZero, replacing heuristic rollouts with a JAX/Flax NNX dual-head neural network for policy priors and value evaluation.
- **`neural_sor/`**: An advanced AlphaZero variant structurally modified to implement Temporal Difference Successive Over-Relaxation (TD-SOR) on the Bellman Operator targets to accelerate convergence.
- **`profiling/`**: A comprehensive benchmarking suite designed to evaluate hardware acceleration performance (CPU vs. Apple Silicon MPS vs. NVIDIA CUDA) across the neural network inference and search tree traversal workloads.
- **`simple/`**: A standalone, single-file minimal working prototype of Tic-Tac-Toe MCTS for pedagogical purposes.

## Execution and Reproducibility

Each directory operates as an independent module with its own dependencies and execution paradigms. Refer to the respective documentation within each directory for installation and execution instructions:

- [Classical MCTS Documentation](mcts/Readme.md)
- [Standard AlphaZero Documentation](neural/Readme.md)
- [TD-SOR AlphaZero Documentation](neural_sor/Readme.md)
- [Hardware Profiling Documentation](profiling/Readme.md)

## Environment Specifications

- **Python Version**: Python 3.9+ recommended.
- **Dependencies**: The foundational `mcts/` directory requires only the Python Standard Library. The `neural/`, `neural_sor/`, and `profiling/` modules depend heavily on `jax`, `flax`, and `optax`. Detailed hardware-specific installation instructions (CPU/GPU/MPS) are provided in their respective `requirements.txt` files.
