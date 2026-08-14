"""
Ring buffer storing self-play examples: (state_array, pi_vector, z).
"""

import random
import numpy as np

class ReplayBuffer:
    """Fixed-capacity ring buffer. Overwrites oldes entries when full. 

    Stores: 
        state_array: np.ndarray of shape (input_size,)
        pi_vector: np.ndarray of shape (num_actions,) - dense policy target
        z: float - game outcome in [-1,1]

    """

    def __init__(self, capacity: int, num_actions: int):
        self.capacity = capacity
        self.num_actions = num_actions
        self._buffer: list = []
        self._cursor = 0 # points to next write position


    def add_game(self, examples: list[tuple], pi_to_vector=None):
        """Add all exmaples from one self-play game. 
        
        Args:
            examples: Output of self_play_game() - list of (state_array, pi_dict, z). 

            pi_to_vector: Optional callable (pi_dict, num_actions) -> np.ndarray. 
                          defaults to built-in _pi_dict_to_vector

        """
        if pi_to_vector is None:
            pi_to_vector = _pi_dict_to_vector

        for state_array, pi_dict, z in examples:
            entry = (
            np.array(state_array),
            pi_to_vector(pi_dict, self.num_actions), 
            float(z),
            )

            if len(self.buffer) < self.capacity:
                self._buffer.append(entry)
            else:
                self._buffer[self._cursor] = entry
            self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random minibatch. 

        Returns:
            states: np.ndarray (batch_size, input_size)
            pi_vecs: np.ndarray (batch_size, num_actions)
            zs: np.ndarray (batch_size,)

        """

        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        states, pi_vecs, zs = zip(*batch)
        return np.stack(states), np.stack(pi_vecs), np.array(zs, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)
    
def _pi_dict_to_vector(pi_dict: dict, num_actions: int) -> np.ndarray:
    """Convert sparse {action: prob} dict to a dense (num_actions,) vector."""
    vec = np.zeros(num_actions, dtype=np.float32)
    for action, prob in pi_dict.items():
        vec[action] = prob
    return vec
