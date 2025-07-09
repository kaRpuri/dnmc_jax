import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
from pathlib import Path
import pickle
import time
import sys
import wandb

# Import config
from config import CONFIG

# Import model and loss
from model import DynamicsModel, compute_loss, prepare_model_inputs

# Import DataLoader
sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
from batch_loader import DataLoader

class TrainState(train_state.TrainState):
    """Train state for Flax models."""
    pass

def create_train_state(rng, learning_rate=1e-3):
    """Initialize model parameters and optimizer."""
    model = DynamicsModel()
    dummy_input = jax.random.normal(rng, (1, 15))  # 15D input
    variables = model.init(rng, dummy_input)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=10000,
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
    """Single training step."""
    def loss_fn(params):
        loss_val, metrics = compute_loss(
            params, state.apply_fn, batch_inputs, batch_targets, 
            beta1, beta2, beta3
        )
        return loss_val, metrics
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss_val, metrics

@jax.jit
def eval_step(state, batch_inputs, batch_targets, beta1=1.0, beta2=0.1, beta3=1e-4):
    """Single evaluation step."""
    loss_val, metrics = compute_loss(
        state.params, state.apply_fn, batch_inputs, batch_targets,
        beta1, beta2, beta3
    )
    return loss_val, metrics

def r2_score(y_true, y_pred):
    """Compute R^2 (coefficient of determination) for regression."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def train_epoch(state, train_loader, beta1, beta2, beta3):
    """Train for one epoch."""
    epoch_losses = []
    epoch_metrics = {'l2_loss': [], 'linf_loss': [], 'l2_reg': []}
    all_targets = []
    all_preds = []
    for batch_inputs, batch_targets in train_loader:
        model_inputs = prepare_model_inputs(batch_inputs)  # Only current features (15D)
        state, loss_val, metrics = train_step(
            state, model_inputs, batch_targets, beta1, beta2, beta3
        )
        preds = np.array(state.apply_fn({'params': state.params}, model_inputs))
        epoch_losses.append(loss_val)
        for key in epoch_metrics:
            epoch_metrics[key].append(metrics[key])
        all_targets.append(np.array(batch_targets))
        all_preds.append(preds)
    avg_loss = jnp.mean(jnp.array(epoch_losses))
    avg_metrics = {k: jnp.mean(jnp.array(v)) for k, v in epoch_metrics.items()}
    # Concatenate all batches for R^2
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    r2 = r2_score(all_targets, all_preds)
    return state, avg_loss, avg_metrics, r2

def validate_epoch(state, val_loader, beta1, beta2, beta3):
    """Validate for one epoch."""
    val_losses = []
    val_metrics = {'l2_loss': [], 'linf_loss': [], 'l2_reg': []}
    all_targets = []
    all_preds = []
    for batch_inputs, batch_targets in val_loader:
        model_inputs = prepare_model_inputs(batch_inputs)
        loss_val, metrics = eval_step(
            state, model_inputs, batch_targets, beta1, beta2, beta3
        )
        preds = np.array(state.apply_fn({'params': state.params}, model_inputs))
        val_losses.append(loss_val)
        for key in val_metrics:
            val_metrics[key].append(metrics[key])
        all_targets.append(np.array(batch_targets))
        all_preds.append(preds)
    avg_loss = jnp.mean(jnp.array(val_losses))
    avg_metrics = {k: jnp.mean(jnp.array(v)) for k, v in val_metrics.items()}
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    r2 = r2_score(all_targets, all_preds)
    return avg_loss, avg_metrics, r2

def train_model(train_data, val_data, config):
    """Main training loop."""
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    beta3 = config['beta3']
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)

    train_loader = DataLoader(
        train_data['inputs'], train_data['outputs'], 
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data['inputs'], val_data['outputs'], 
        batch_size=batch_size, shuffle=False
    )

    rng = jax.random.PRNGKey(config.get('seed', 42))
    state = create_train_state(rng, learning_rate)
    best_val_loss = float('inf')

    print(f"Starting training for {epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(epochs):
        start_time = time.time()
        state, train_loss, train_metrics, train_r2 = train_epoch(
            state, train_loader, beta1, beta2, beta3
        )
        val_loss, val_metrics, val_r2 = validate_epoch(
            state, val_loader, beta1, beta2, beta3
        )
        epoch_time = time.time() - start_time

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': float(train_loss),
            'val/loss': float(val_loss),
            'train/l2_loss': float(train_metrics['l2_loss']),
            'val/l2_loss': float(val_metrics['l2_loss']),
            'train/linf_loss': float(train_metrics['linf_loss']),
            'val/linf_loss': float(val_metrics['linf_loss']),
            'train/l2_reg': float(train_metrics['l2_reg']),
            'val/l2_reg': float(val_metrics['l2_reg']),
            'train/r2': float(train_r2),
            'val/r2': float(val_r2),
            'epoch_time': epoch_time
        }, step=epoch)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"         | "
                  f"Train L2: {train_metrics['l2_loss']:.6f} | "
                  f"Val L2: {val_metrics['l2_loss']:.6f}")

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
            wandb.run.summary['best_val_loss'] = float(val_loss)

    print("\nTraining completed!")
    return state, best_val_loss

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project=CONFIG['project'],
        entity=CONFIG['entity'],
        name=CONFIG['run_name'],
        config=CONFIG
    )

    # Load normalized data
    train_data = np.load(CONFIG['train_data_path'])
    val_data = np.load(CONFIG['val_data_path'])

    # Train the model
    final_state, best_loss = train_model(train_data, val_data, CONFIG)
    print(f"Best validation loss: {best_loss:.6f}")
    wandb.finish()
