import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class DataLoader:
    """
    Simple data loader for batching time-series data without breaking temporal order.
    """
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, batch_size: int=1024, shuffle: bool=False):
        """
        Args:
            inputs: Array of shape (N, F_in) containing input features [1].
            targets: Array of shape (N, F_out) containing target derivatives [1].
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle indices each epoch.
        """
        assert len(inputs) == len(targets), "Inputs and targets must have same length"  # [1]
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(inputs))
        self._reset_epoch()

    def _reset_epoch(self):
        "Resets and shuffles indices at the start of each epoch."  
        if self.shuffle:
            np.random.shuffle(self.indices)  # [1]
        self.current = 0

    def __iter__(self):
        "Returns the iterator object itself."  
        self._reset_epoch()
        return self

    def __next__(self):
        "Yields the next batch of (inputs, targets) as JAX arrays."  
        if self.current >= len(self.indices):
            raise StopIteration
        idx = self.indices[self.current : self.current + self.batch_size]
        batch_in = jnp.asarray(self.inputs[idx])   # [1]
        batch_out = jnp.asarray(self.targets[idx]) # [1]
        self.current += self.batch_size
        return batch_in, batch_out

    def __len__(self):
        "Returns the total number of batches per epoch."  
        return (len(self.inputs) + self.batch_size - 1) // self.batch_size  # [1]




