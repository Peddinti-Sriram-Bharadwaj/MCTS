# Monte Carlo Tree Search (MCTS)

A clean Python implementation of Monte Carlo Tree Search for two-player zero-sum board games.

## Repository Structure

- **`mcts/`**: Modular MCTS engine built using the Strategy Pattern (Dependency Injection). Supports Tic-Tac-Toe and Connect-4.
- **`simple/`**: Standalone single-file prototype of Tic-Tac-Toe MCTS.

## Environment & Requirements

- **Python Version**: Python 3.7+
- **Dependencies**: None (Uses pure Python standard library: `abc`, `dataclasses`, `argparse`, `math`, `random`, `copy`, `typing`)

## Reproducibility & Execution

Run self-play games using the modular framework:

```bash
# Play Tic-Tac-Toe (500 iterations per move)
python3 mcts/play.py --game tictactoe

# Play Connect-4 (1000 iterations per move)
python3 mcts/play.py --game connect4
```

For full architecture details and instructions on plugging in new games, see [mcts/Readme.md](mcts/Readme.md).
