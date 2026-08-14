"""
Training step and self-play training loop.
Loss = policy cross-entropy + value MSE.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

from network.model import AlphaZeroNet
from network.inference import make_network_fn
from training.replay_buffer import ReplayBuffer
from training.self_play import self_play_game
from training.checkpoint import save_checkpoint


def loss_fn(
    model: AlphaZeroNet,
    states: jnp.ndarray,
    pi_targets: jnp.ndarray,
    z_targets: jnp.ndarray,
) -> jnp.ndarray:
    """Combined AlphaZero loss over a minibatch.

    policy loss: cross-entropy between MCTS pi and network policy output.
    value  loss: MSE between game outcome z and network value output.

    Args:
        states:     (batch, input_size)
        pi_targets: (batch, num_actions) — MCTS visit-count distributions
        z_targets:  (batch,) — game outcomes in [-1, 1]

    Returns:
        Scalar loss.
    """
    # nnx.vmap applies model to each example in the batch independently,
    # correctly handling NNX module state.
    batched_forward = nnx.vmap(model, in_axes=0, out_axes=0)
    policy_logits, values = batched_forward(states)  # (batch, num_actions), (batch, 1)

    values = values.squeeze(-1)  # (batch,)

    # Policy loss: cross-entropy. pi_targets is already a probability distribution.
    # log_softmax is numerically more stable than log(softmax(logits)).
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    policy_loss = -jnp.sum(pi_targets * log_probs, axis=-1).mean()

    # Value loss: MSE between predicted value and actual outcome.
    value_loss = jnp.mean((values - z_targets) ** 2)

    return policy_loss + value_loss


def train_step(
    model: AlphaZeroNet,
    optimizer: nnx.Optimizer,
    states: np.ndarray,
    pi_targets: np.ndarray,
    z_targets: np.ndarray,
) -> float:
    """One gradient update step. Returns the scalar loss value."""
    states     = jnp.array(states)
    pi_targets = jnp.array(pi_targets)
    z_targets  = jnp.array(z_targets)

    loss, grads = nnx.value_and_grad(loss_fn)(model, states, pi_targets, z_targets)
    optimizer.update(model, grads)

    return float(loss)


def train_loop(
    model: AlphaZeroNet,
    initial_state_fn: Callable,
    num_iterations: int = 50,
    games_per_iteration: int = 5,
    num_simulations: int = 50,
    batch_size: int = 64,
    buffer_capacity: int = 5000,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 10,
):
    """AlphaZero self-play training loop.

    Each iteration:
        1. Play `games_per_iteration` self-play games to fill the buffer.
        2. Sample a minibatch and do one gradient step.

    Args:
        model:               AlphaZeroNet instance.
        initial_state_fn:    Callable () -> GameState. e.g. TicTacToe.initial
        num_iterations:      Number of self-play + train cycles.
        games_per_iteration: Self-play games generated per iteration.
        num_simulations:     MCTS simulations per move during self-play.
        batch_size:          Minibatch size for training.
        buffer_capacity:     Max examples in replay buffer.
        learning_rate:       Adam learning rate.
        checkpoint_dir:      Directory to save checkpoints in.
        checkpoint_every:    Save a checkpoint every N training iterations.
    """
    sample_state = initial_state_fn()
    num_actions = sample_state.num_actions

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    buffer = ReplayBuffer(capacity=buffer_capacity, num_actions=num_actions)

    for iteration in range(num_iterations):
        # --- Self-play phase ---
        network_fn = make_network_fn(model)
        for _ in range(games_per_iteration):
            examples = self_play_game(initial_state_fn(), network_fn, num_simulations)
            buffer.add_game(examples)

        if len(buffer) < batch_size:
            print(f"[iter {iteration}] buffer too small ({len(buffer)}), skipping train step.")
            continue

        # --- Training phase ---
        states, pi_vecs, zs = buffer.sample(batch_size)
        loss = train_step(model, optimizer, states, pi_vecs, zs)

        print(f"[iter {iteration:3d}] loss={loss:.4f}  buffer={len(buffer)}")

        # --- Checkpoint phase ---
        if iteration % checkpoint_every == 0:
            save_checkpoint(model, checkpoint_dir, iteration)
