"""
Alphazero-style neural network: shared trunk + policy head + value head. 
Built with Flax NNX. 
"""

import jax.numpy as jnp
from flax import nnx

class AlphazeroNet(nnx.Module):
    """Shared-trunk with two heads.

    Args:
        input_size: Flattened size of the game state array. 
        num_actions: Size of the policy head output
        rngs: Flax NNX random state

    """

    def __init__(
    self, 
    input_size: int, 
    num_actions: int, 
    hidden_size: int=64,
    rngs: nnx.Rngs = None,
    ):
        # Shared trunk: two dense layers with ReLU
        self.trunk1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.trunk2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Policy head: hidden -> num_actions (raw logits, softmax applied later)
        self.policy_head = nnx.Linear(hidden_size, num_actions, rngs=rngs)

        # Value head: hidden -> 1 scalar, then tanh to constrain to [-1, 1]
        self.value_head = nnx.Linear(hidden_size, 1, rngs=rngs)


    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass. 
        
        Args:
            x : Game state array, shape (input_size,) - must be pre-flattened

    
        Returns:
            policy_logits: Raw logits, shape (num_actions,). Apply softmax + mask externally. 
            
            value: Scalar estimate in [-1, 1], shape (1,). 

        """
        #Shared trunk

        x = nnx.relu(self.trunk1(x))
        x = nnx.relu(self.trunk2(x))

        # Heads
        policy_logits = self.policy_head(x)
        value = jnp.tanh(self.value(head(x))

        return policy_logits, value


    
