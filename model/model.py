import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import yaml
from pathlib import Path

# Load configuration from config.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

class DynamicsModel(nn.Module):
    """Enhanced dynamics model with residual connections and better layer sizing"""
    
    @nn.compact
    def __call__(self, x, training=False):
        # Input normalization layer (helps training stability)
        x_norm = nn.LayerNorm()(x)
        
        h1 = nn.Dense(CONFIG['model']['hidden_dim'])(x_norm)
        h1 = nn.relu(h1)
        h1 = nn.Dropout(CONFIG['model']['dropout_rate'], deterministic=not training)(h1)
        
        h2 = nn.Dense(CONFIG['model']['hidden_dim'] * 2)(h1)
        h2 = nn.LayerNorm()(h2)
        h2 = nn.relu(h2)
        h2 = nn.Dense(CONFIG['model']['hidden_dim'] * 2)(h2)
        h2 = nn.LayerNorm()(h2)
        h2 = nn.relu(h2)
        h2 = h2 + nn.Dense(CONFIG['model']['hidden_dim'] * 2)(h1)  # Skip connection
        
        h3 = nn.Dense(CONFIG['model']['hidden_dim'] * 2)(h2)
        h3 = nn.LayerNorm()(h3)
        h3 = nn.relu(h3)
        h3 = nn.Dense(CONFIG['model']['hidden_dim'] * 2)(h3)
        h3 = nn.LayerNorm()(h3)
        h3 = nn.relu(h3)
        h3 = h3 + h2  # Skip connection
        
        # Output layers with separate paths for different output types
        # Position changes
        pos_out = nn.Dense(CONFIG['model']['output_dim'])(h3)
        pos_out = nn.relu(pos_out)
        pos_out = nn.Dense(2)(pos_out)
        
        # Steering change (sensitive to command inputs)
        steer_out = nn.Dense(CONFIG['model']['output_dim'])(jnp.concatenate([h3, x[:, 13:15]], axis=1))  # Add raw commands
        steer_out = nn.relu(steer_out)
        steer_out = nn.Dense(1)(steer_out)
        
        # Velocity changes
        vel_out = nn.Dense(CONFIG['model']['output_dim'])(h3)
        vel_out = nn.relu(vel_out)
        vel_out = nn.Dense(2)(vel_out)
        
        # Angular changes
        ang_out = nn.Dense(CONFIG['model']['output_dim'])(h3)
        ang_out = nn.relu(ang_out)
        ang_out = nn.Dense(2)(ang_out)
        
        # Combine all outputs
        return jnp.concatenate([pos_out, steer_out, vel_out, ang_out], axis=1)

def prepare_model_inputs(current_features):
    """
    Use only current features (15D) for model input.
    
    Args:
        current_features: (batch_size, 15) current vehicle state features
        
    Returns:
        model_inputs: (batch_size, 15) input for the model
    """
    return current_features

def create_model():
    """
    Factory function to create the vehicle dynamics model.
    
    Returns:
        VehicleDynamicsModel instance
    """
    return DynamicsModel()


def compute_loss(params, model, inputs, targets, beta1=CONFIG['loss_weights']['beta1'], beta2=CONFIG['loss_weights']['beta2'], beta3=CONFIG['loss_weights']['beta3']):
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

    # 2. Create dummy input batch (batch_size=4, input_dim=15)
    dummy_inputs = jax.random.normal(key, (4, 15))
    
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
