"""
AlphaZero neural network with standard architecture.
TD-SOR is now applied to the targets during training, not structurally.

Shared trunk: Standard Linear -> ReLU -> Linear -> ReLU
Policy Head:  Standard Linear -> Logits
Value Head:   Standard Linear -> tanh
"""

import jax.numpy as jnp
from flax import nnx

class AlphaZeroNet(nnx.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_size: int = 64,
        # omega is kept here strictly for metadata tracking, it's not used in forward pass
        omega: float = 1.25,  
        *,
        rngs: nnx.Rngs,
    ):
        self.omega = omega

        # Standard Shared Trunk
        self.fc1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Standard Policy Head
        self.policy_head = nnx.Linear(hidden_size, num_actions, rngs=rngs)

        # Standard Value Head
        self.value_head = nnx.Linear(hidden_size, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Flattened state array, shape (input_size,) or (batch, input_size).

        Returns:
            policy_logits: (num_actions,) unnormalized policy logits.
            value:         (1,) scalar in [-1, 1] from value head.
        """
        if x.ndim == 3:
            # Flatten 2D planes into 1D (per batch)
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 2 and x.shape[0] == 2:
            # Flatten single 3D array (2, H, W) to 1D
            x = x.flatten()

        # Shared Trunk
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))

        # Policy Output
        policy_logits = self.policy_head(x)

        # Value Output
        value = jnp.tanh(self.value_head(x))

        return policy_logits, value
