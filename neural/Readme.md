# MCTS: Pluggable Game Environments

A refactored Monte Carlo Tree Search implementation using the **Strategy pattern** (dependency injection). Swap games by passing a `--game` flag at runtime.

## Requirements & Reproducibility

- **Python Version**: Python 3.7+
- **Dependencies**: No external packages required (uses pure standard library: `abc`, `dataclasses`, `argparse`, `math`, `random`, `copy`, `typing`).
- **Reproducibility**: MCTS rollout simulations rely on Python's standard `random` module. For deterministic evaluations across runs, set a fixed random seed in python (e.g. `import random; random.seed(42)`).

## Architecture

### Abstract Interface
- **`game_state_abstract.py`** — `GameState` (ABC): defines the contract any game must implement.
  - Properties/methods: `current_player`, `legal_actions()`, `apply_action()`, `is_terminal()`, `reward()`, `render()`
  - No game-specific logic; pure interface.

### Concrete Games
- **`tictactoe.py`** — Tic-tac-toe (9-cell board, 3×3 grid)
- **`connect4.py`** — Connect-4 (7×6 board, drop-to-gravity mechanics)

Both inherit from `GameState` and implement the full interface. Zero changes needed to MCTS itself when adding a new game.

### MCTS Core (Game-Agnostic)
- **`node.py`** — Tree node structure (visit count, value, children dict)
- **`selection.py`** — UCB1 tree traversal with negated child values (perspective flip fix)
- **`expansion.py`** — Lazy one-child-at-a-time expansion
- **`simulation.py`** — Random rollout with perspective tracking
- **`backprop.py`** — Path update with sign alternation
- **`search.py`** — Main search loop, coordinates all pieces

These six files are **completely game-agnostic**. They depend only on the `GameState` abstract interface.

### Driver
- **`play.py`** — Self-play driver with `--game` flag selection
  - Accepts `--game tictactoe` or `--game connect4`
  - Prints board state after each move
  - Reports winner or draw at the end

## Usage

```bash
# Play tic-tac-toe (500 iterations per move)
python3 play.py --game tictactoe

# Play Connect-4 (1000 iterations per move)
python3 play.py --game connect4
```

## How to Add a New Game

1. Create a file `my_game.py` with a class inheriting from `GameState`:
   ```python
   from game_state_abstract import GameState
   
   class MyGame(GameState):
       @property
       def current_player(self) -> str:
           # ...
       
       def legal_actions(self) -> list:
           # ...
       
       # ... implement all abstract methods
   ```

2. Add it to `play.py`'s `GAMES` dict:
   ```python
   GAMES = {
       "tictactoe": TicTacToe,
       "connect4": Connect4,
       "mygame": MyGame,  # <- add this
   }
   ```

3. Run: `python3 play.py --game mygame`

**No changes to MCTS itself are needed.** The search algorithm works with any game that implements the `GameState` interface.

## Key Design Patterns

### Strategy Pattern (Dependency Injection)
- Games are swappable at runtime via a flag
- MCTS depends only on the abstract interface, not concrete games
- New games require no MCTS modifications

### Perspective Flip Bug (Lesson from Integration Testing)
- `backprop` stores values from the child node's current_player perspective
- `selection` must negate the child's stored value when scoring, because the parent is making the choice (opponent's turn relative to the child)
- This bug was only caught by full self-play testing, not unit-level checks
- See `selection.py` line in `ucb_score()`: `exploitation = -(child.total_value / child.visit_count)`

## Expected Outcomes

- **Tic-tac-toe self-play**: Always draws (perfect game, both sides play perfectly)
- **Connect-4 self-play**: Varies; with 1000 iterations per move, roughly 50/50 win rates or frequent draws depending on first-move advantage and board complexity

## File Dependency Graph

```
play.py
  ├── search.py
  │   ├── selection.py
  │   │   └── node.py
  │   ├── expansion.py
  │   │   └── node.py
  │   ├── simulation.py
  │   │   └── game_state_abstract.py
  │   └── backprop.py
  │       └── node.py
  ├── tictactoe.py
  │   └── game_state_abstract.py
  └── connect4.py
      └── game_state_abstract.py
```

All paths flow through the abstract `GameState` interface. No circular dependencies.
