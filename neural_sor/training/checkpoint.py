"""
Checkpoint utilities using orbax-checkpoint.
Saves/loads AlphaZeroNet weights to/from indexed directories.

Directory layout:
    checkpoints/
        ckpt_00000/   ← iteration 0
        ckpt_00010/   ← iteration 10
        ...
"""

import re
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from network.model import AlphaZeroNet

CKPT_PREFIX = "ckpt_"
_PATTERN = re.compile(rf"^{re.escape(CKPT_PREFIX)}(\d+)$")


def _ckpt_path(checkpoint_dir: str, index: int) -> Path:
    return (Path(checkpoint_dir) / f"{CKPT_PREFIX}{index:05d}").resolve()


def save_checkpoint(model: AlphaZeroNet, checkpoint_dir: str, index: int) -> None:
    """Save model weights to checkpoint_dir/ckpt_{index:05d}/."""
    path = _ckpt_path(checkpoint_dir, index)
    path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    state = nnx.state(model)
    checkpointer.save(str(path), state, force=True)
    checkpointer.wait_until_finished()
    print(f"  [ckpt] saved  → {path}")


def load_checkpoint(
    model: AlphaZeroNet,
    checkpoint_dir: str,
    index: int | None = None,
) -> int:
    """Load weights into model in-place from checkpoint_dir.

    Args:
        model:          AlphaZeroNet instance with matching architecture.
        checkpoint_dir: Root directory containing ckpt_NNNNN/ subdirs.
        index:          Specific checkpoint index. If None, loads the latest.

    Returns:
        The index that was loaded.

    Raises:
        FileNotFoundError: If no checkpoints exist or requested index is missing.
    """
    if index is None:
        index = latest_checkpoint_index(checkpoint_dir)
        if index is None:
            raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}'")

    path = _ckpt_path(checkpoint_dir, index)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpointer = ocp.StandardCheckpointer()
    abstract_state = nnx.state(model)
    restored = checkpointer.restore(str(path), target=abstract_state)
    nnx.update(model, restored)
    print(f"  [ckpt] loaded ← {path}")
    return index


def latest_checkpoint_index(checkpoint_dir: str) -> int | None:
    """Return the highest checkpoint index in checkpoint_dir, or None if empty."""
    ckpt_dir = Path(checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        return None

    indices = [
        int(m.group(1))
        for entry in ckpt_dir.iterdir()
        if (m := _PATTERN.match(entry.name)) and entry.is_dir()
    ]
    return max(indices) if indices else None


def list_checkpoints(checkpoint_dir: str) -> list[int]:
    """Return a sorted list of all available checkpoint indices."""
    ckpt_dir = Path(checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        return []

    return sorted(
        int(m.group(1))
        for entry in ckpt_dir.iterdir()
        if (m := _PATTERN.match(entry.name)) and entry.is_dir()
    )
