"""
Training step and self-play training loop with True TD-SOR Loss.
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
    q_targets: jnp.ndarray,
    omega_target: float
) -> jnp.ndarray:
    """Combined AlphaZero loss with True TD-SOR Target Relaxation.

    Args:
        states:     (batch, input_size)
        pi_targets: (batch, num_actions) — MCTS visit-count distributions
        q_targets:  (batch,) — max Q-value from the MCTS root
        omega_target: Relaxation parameter for SOR loss

    Returns:
        Scalar total loss (Policy Loss + Value SOR Loss).
    """
    batched_forward = nnx.vmap(model, in_axes=0, out_axes=0)
    policy_logits, values = batched_forward(states)

    values = values.squeeze(-1)  # (batch,)

    # 1. Policy Loss: cross-entropy
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    policy_loss = -jnp.sum(pi_targets * log_probs, axis=-1).mean()

    # 2. True TD-SOR Loss (Kamanchi et al.):
    # V_target = (1 - omega) * V_old + omega * max_Q
    v_detached = jax.lax.stop_gradient(values)
    sor_target = (1.0 - omega_target) * v_detached + omega_target * q_targets
    value_loss = jnp.mean((values - sor_target) ** 2)

    return policy_loss + value_loss


def train_step(
    model: AlphaZeroNet,
    optimizer: nnx.Optimizer,
    states: np.ndarray,
    pi_targets: np.ndarray,
    q_targets: np.ndarray,
    omega_target: float
) -> float:
    """One gradient update step. Returns the scalar loss value."""
    states     = jnp.array(states)
    pi_targets = jnp.array(pi_targets)
    q_targets  = jnp.array(q_targets)

    loss_fn_with_omega = lambda m, s, p, q: loss_fn(m, s, p, q, omega_target)
    loss, grads = nnx.value_and_grad(loss_fn_with_omega)(model, states, pi_targets, q_targets)
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
    omega_target: float = 1.25
):
    """TD-SOR self-play training loop."""
    sample_state = initial_state_fn()
    num_actions = sample_state.num_actions

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    buffer = ReplayBuffer(capacity=buffer_capacity, num_actions=num_actions)

    for iteration in range(num_iterations):
        network_fn = make_network_fn(model)
        for _ in range(games_per_iteration):
            examples = self_play_game(initial_state_fn(), network_fn, num_simulations)
            buffer.add_game(examples)

        if len(buffer) < batch_size:
            print(f"[iter {iteration}] buffer too small ({len(buffer)}), skipping train step.")
            continue

        states, pi_vecs, qs = buffer.sample(batch_size)
        loss = train_step(model, optimizer, states, pi_vecs, qs, omega_target)

        print(f"[iter {iteration:3d}] loss={loss:.4f}  buffer={len(buffer)}")

        if iteration % checkpoint_every == 0:
            save_checkpoint(model, checkpoint_dir, iteration)
