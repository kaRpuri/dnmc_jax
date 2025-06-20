import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
from pathlib import Path
import pickle
import time

# Import your existing modules
from model import DynamicsModel, compute_loss


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
from batch_loader import DataLoader

class TrainState(train_state.TrainState):
    """Extended TrainState to include batch normalization statistics if needed."""
    pass

def create_train_state(rng, learning_rate=1e-3):
    """Initialize model parameters and optimizer."""
    model = DynamicsModel()
    dummy_input = jax.random.normal(rng, (1, 22))
    variables = model.init(rng, dummy_input)
    
    # Create optimizer with learning rate schedule
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=10000,  # Adjust based on your dataset size
        alpha=0.1
    )
    optimizer = optax.adam(schedule)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

@jax.jit
def train_step(state, batch_inputs, batch_targets, beta1=1.0, beta2=0.1, beta3=1e-4):
    """Perform one training step with gradient computation and parameter update."""
    
    def loss_fn(params):
        loss_val, metrics = compute_loss(
            params, state.apply_fn, batch_inputs, batch_targets, 
            beta1, beta2, beta3
        )
        return loss_val, metrics
    
    # Compute gradients
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss_val, metrics

@jax.jit
def eval_step(state, batch_inputs, batch_targets, beta1=1.0, beta2=0.1, beta3=1e-4):
    """Evaluate model on validation batch without parameter updates."""
    loss_val, metrics = compute_loss(
        state.params, state.apply_fn, batch_inputs, batch_targets,
        beta1, beta2, beta3
    )
    return loss_val, metrics

def prepare_model_inputs(current_features, previous_states=None):
    """Combine current features (15D) with previous delta states (7D)."""
    batch_size = current_features.shape[0]
    
    # Use zeros for previous states if not provided (first timestep)
    if previous_states is None:
        previous_states = jnp.zeros((batch_size, 7))
    
    return jnp.concatenate([current_features, previous_states], axis=1)

def train_epoch(state, train_loader, beta1, beta2, beta3):
    """Train for one epoch and return average metrics."""
    epoch_losses = []
    epoch_metrics = {'l2_loss': [], 'linf_loss': [], 'l2_reg': []}
    
    for batch_inputs, batch_targets in train_loader:
        # Prepare inputs with previous states (simplified: use zeros)
        model_inputs = prepare_model_inputs(batch_inputs)
        
        # Training step
        state, loss_val, metrics = train_step(
            state, model_inputs, batch_targets, beta1, beta2, beta3
        )
        
        # Collect metrics
        epoch_losses.append(loss_val)
        for key in epoch_metrics:
            epoch_metrics[key].append(metrics[key])
    
    # Average metrics over epoch
    avg_loss = jnp.mean(jnp.array(epoch_losses))
    avg_metrics = {k: jnp.mean(jnp.array(v)) for k, v in epoch_metrics.items()}
    
    return state, avg_loss, avg_metrics

def validate_epoch(state, val_loader, beta1, beta2, beta3):
    """Validate for one epoch and return average metrics."""
    val_losses = []
    val_metrics = {'l2_loss': [], 'linf_loss': [], 'l2_reg': []}
    
    for batch_inputs, batch_targets in val_loader:
        # Prepare inputs
        model_inputs = prepare_model_inputs(batch_inputs)
        
        # Validation step
        loss_val, metrics = eval_step(
            state, model_inputs, batch_targets, beta1, beta2, beta3
        )
        
        # Collect metrics
        val_losses.append(loss_val)
        for key in val_metrics:
            val_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_loss = jnp.mean(jnp.array(val_losses))
    avg_metrics = {k: jnp.mean(jnp.array(v)) for k, v in val_metrics.items()}
    
    return avg_loss, avg_metrics

def train_model(train_data, val_data, config):
    """Main training function."""
    
    # Unpack configuration
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 1e-3)
    beta1 = config.get('beta1', 1.0)
    beta2 = config.get('beta2', 0.1)
    beta3 = config.get('beta3', 1e-4)
    save_dir = Path(config.get('save_dir', './checkpoints'))
    
    # Create save directory
    save_dir.mkdir(exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data['inputs'], train_data['outputs'], 
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data['inputs'], val_data['outputs'], 
        batch_size=batch_size, shuffle=False
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        state, train_loss, train_metrics = train_epoch(
            state, train_loader, beta1, beta2, beta3
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(
            state, val_loader, beta1, beta2, beta3
        )
        
        epoch_time = time.time() - start_time
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"         | "
                  f"Train L2: {train_metrics['l2_loss']:.6f} | "
                  f"Val L2: {val_metrics['l2_loss']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'params': state.params,
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config
            }
            with open(save_dir / 'best_model.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"         | New best model saved! Val loss: {val_loss:.6f}")
    
    print("\nTraining completed!")
    return state, best_val_loss

# Example usage
if __name__ == "__main__":
    # Load your preprocessed data
    train_data = np.load('../data_processing/processed/train.npz')
    val_data = np.load('../data_processing/processed/val.npz')
    
    # Training configuration
    config = {
        'batch_size': 256,
        'epochs': 100,
        'learning_rate': 1e-3,
        'beta1': 1.0,    # L2 prediction loss weight
        'beta2': 0.1,    # L∞ reconstruction loss weight
        'beta3': 1e-4,   # L2 regularization weight
        'save_dir': './checkpoints'
    }
    
    # Train the model
    final_state, best_loss = train_model(train_data, val_data, config)
    
    print(f"Best validation loss: {best_loss:.6f}")
