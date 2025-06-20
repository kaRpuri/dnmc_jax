import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class DynamicsModel(nn.Module):
    """
    Multi-layer perceptron for vehicle dynamics delta state prediction.
    
    Architecture: [22, 32, 64, 128, 128, 64, 32, 7]
    - Input: 22D (15 current features + 7 previous delta states)
    - Output: 7D delta states [dx, dy, d_steering, dvx, dvy, dwz, dyaw]
    """
    
    @nn.compact
    def __call__(self, x):
        # Layer 1: 22 -> 32
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # Layer 2: 32 -> 64
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Layer 3: 64 -> 128
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Layer 4: 128 -> 128 (same size)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Layer 5: 128 -> 64
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Layer 6: 64 -> 32
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # Output layer: 32 -> 7 (no activation for regression)
        x = nn.Dense(7)(x)
        
        return x

def prepare_model_inputs(current_features, previous_delta_states):
    """
    Combine current features (15D) with previous delta states (7D) for model input.
    
    Args:
        current_features: (batch_size, 15) current vehicle state features
        previous_delta_states: (batch_size, 7) previous delta state values
        
    Returns:
        model_inputs: (batch_size, 22) concatenated input for the model
    """
    return jnp.concatenate([current_features, previous_delta_states], axis=1)

def create_model():
    """
    Factory function to create the vehicle dynamics model.
    
    Returns:
        VehicleDynamicsModel instance
    """
    return DynamicsModel()


def compute_loss(params, model, inputs, targets, beta1=1.0, beta2=0.1, beta3=1e-4):
    # Change this line:
    # preds = model.apply({'params': params}, inputs)
    
    # To this:
    preds = model({'params': params}, inputs)
    
    # Rest of the function remains the same
    l2_loss = jnp.mean(jnp.sum((targets - preds) ** 2, axis=-1))
    linf_loss = jnp.mean(jnp.max(jnp.abs(targets - preds), axis=-1))
    l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    total_loss = beta1 * l2_loss + beta2 * linf_loss + beta3 * l2_reg
    metrics = {'l2_loss': l2_loss, 'linf_loss': linf_loss, 'l2_reg': l2_reg, 'total_loss': total_loss}
    return total_loss, metrics



if __name__ == "__main__":
    # 1. Instantiate model and RNG
    model = create_model()
    key   = jax.random.PRNGKey(42)

    # 2. Create dummy input batch (batch_size=4, input_dim=22)
    dummy_inputs = jax.random.normal(key, (4, 22))
    
    # 3. Initialize model parameters
    init_vars = model.init(key, dummy_inputs)
    params    = init_vars['params']
    
    # 4. Forward-pass test
    outputs = model.apply({'params': params}, dummy_inputs)
    print("Model Architecture Test:")
    print(f"  Input shape:  {dummy_inputs.shape}")
    print(f"  Output shape: {outputs.shape} (expected (4, 7))")
    
    # 5. Parameter-shape verification
    print("\nParameter shapes:")
    for layer, p in params.items():
        k_shape = p['kernel'].shape
        b_shape = p['bias'].shape
        print(f"  {layer}: kernel {k_shape}, bias {b_shape}")
    
    # 6. Dummy targets for loss computation
    dummy_targets = jax.random.normal(key, (4, 7))
    
    # 7. Compute loss and display metrics
    # Pass model.apply (the function) instead of model (the object)
    loss_val, metrics = compute_loss(params, model.apply, dummy_inputs, dummy_targets)
    print("\nLoss Function Test:")
    print(f"  Total loss: {loss_val:.6f}")
    print(f"  L2 loss:    {metrics['l2_loss']:.6f}")
    print(f"  L∞ loss:    {metrics['linf_loss']:.6f}")
    print(f"  L2 reg:     {metrics['l2_reg']:.6f}")
