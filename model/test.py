import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import yaml
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import model components
from model import DynamicsModel, prepare_model_inputs
from data_processing.preprocessor import Preprocessor

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = DynamicsModel()
    params = checkpoint['params']
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.6f}")
    return model, params

def r2_score(y_true, y_pred):
    """Compute R^2 (coefficient of determination) for regression."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def compute_metrics(true, pred):
    """Compute evaluation metrics between true and predicted values."""
    # L2 loss (MSE)
    l2_loss = np.mean(np.sum((true - pred) ** 2, axis=-1))
    
    # L-infinity loss (maximum absolute error)
    linf_loss = np.mean(np.max(np.abs(true - pred), axis=-1))
    
    # R2 score (overall)
    overall_r2 = r2_score(true, pred)
    
    # Per-component R2 scores
    component_r2 = {
        'pos_x': r2_score(true[:, 0], pred[:, 0]),
        'pos_y': r2_score(true[:, 1], pred[:, 1]),
        'steering': r2_score(true[:, 2], pred[:, 2]),
        'vel_x': r2_score(true[:, 3], pred[:, 3]),
        'vel_y': r2_score(true[:, 4], pred[:, 4]),
        'ang_vel': r2_score(true[:, 5], pred[:, 5]),
        'heading': r2_score(true[:, 6], pred[:, 6])
    }
    
    # RMSE overall and per component
    rmse = np.sqrt(np.mean((true - pred) ** 2, axis=0))
    rmse_overall = np.sqrt(np.mean((true - pred) ** 2))
    
    # Max absolute error per component
    max_abs_error = np.max(np.abs(true - pred), axis=0)
    
    metrics = {
        'l2_loss': l2_loss,
        'linf_loss': linf_loss,
        'r2_score': overall_r2,
        'component_r2': component_r2,
        'rmse': rmse,
        'rmse_overall': rmse_overall,
        'max_abs_error': max_abs_error
    }
    
    return metrics

def plot_prediction_vs_actual(true_deltas, pred_deltas):
    """Plot predicted vs actual state derivatives."""
    component_names = [
        'Position X Change', 'Position Y Change', 'Steering Change',
        'Velocity X Change', 'Velocity Y Change', 'Angular Vel Change', 'Heading Change'
    ]
    
    plt.figure(figsize=(18, 10))
    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.scatter(true_deltas[:, i], pred_deltas[:, i], alpha=0.3, s=2)
        
        # Add diagonal line for perfect prediction
        min_val = min(true_deltas[:, i].min(), pred_deltas[:, i].min())
        max_val = max(true_deltas[:, i].max(), pred_deltas[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'{component_names[i]}')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_plots/prediction_vs_actual.png')
    plt.close()

def plot_error_distribution(true_deltas, pred_deltas):
    """Plot error distribution for each component."""
    errors = pred_deltas - true_deltas
    component_names = [
        'Position X', 'Position Y', 'Steering',
        'Velocity X', 'Velocity Y', 'Angular Velocity', 'Heading'
    ]
    
    plt.figure(figsize=(18, 10))
    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.hist(errors[:, i], bins=50, alpha=0.7)
        plt.title(f'{component_names[i]} Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_plots/error_distribution.png')
    plt.close()

def simulate_trajectory(model, params, initial_state, dt=0.1, steps=100):
    """Simulate a trajectory with the learned model."""
    trajectory = [initial_state[:2].copy()]  # Store only x,y positions
    current_state = initial_state.copy()
    
    for i in range(steps):
        # Prepare model input
        model_input = prepare_model_inputs(current_state.reshape(1, -1))
        
        # Get model prediction (state derivatives)
        state_dot = model.apply({'params': params}, model_input)[0]
        
        # Update state with Euler integration
        current_state[0:2] += state_dot[3:5] * dt  # Velocity update
        current_state[5] += state_dot[5] * dt      # Angular velocity update
        current_state[14] += state_dot[2] * dt     # Steering update
        
        # Add position changes to trajectory
        position = trajectory[-1] + state_dot[0:2] * dt
        trajectory.append(position)
    
    return np.array(trajectory)

def plot_trajectory_comparison(test_data, true_deltas, pred_deltas, num_samples=3):
    """Plot actual vs predicted trajectories for selected test segments."""
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 12))
    
    # Get random segments
    for i in range(num_samples):
        start_idx = np.random.randint(0, len(test_data['positions']) - 200)
        segment_length = 200
        
        # Get actual positions from test data
        actual_positions = test_data['positions'][start_idx:start_idx+segment_length]
        
        # Initialize predicted trajectory
        integrated_positions = [actual_positions[0]]
        
        # Timestamps for dt calculation
        timestamps = test_data['timestamps'][start_idx:start_idx+segment_length]
        dt = np.diff(timestamps)
        
        # Integrate predicted position changes
        for j in range(segment_length - 1):
            dx_dy_pred = pred_deltas[start_idx + j, 0:2]
            next_pos = integrated_positions[-1] + dx_dy_pred * dt[j]
            integrated_positions.append(next_pos)
        
        integrated_positions = np.array(integrated_positions)
        
        # Plot
        axes[i].plot(actual_positions[:, 0], actual_positions[:, 1], 'b-', label='Actual')
        axes[i].plot(integrated_positions[:, 0], integrated_positions[:, 1], 'r--', label='Predicted')
        axes[i].set_title(f'Trajectory Segment {i+1}')
        axes[i].set_xlabel('X Position (m)')
        axes[i].set_ylabel('Y Position (m)')
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('test_plots/trajectory_comparison.png')
    plt.close()

def plot_error_heatmap(test_data, true_deltas, pred_deltas):
    """Plot prediction error as a heatmap on the trajectory."""
    # Calculate overall error for each data point
    errors = np.mean(np.abs(pred_deltas - true_deltas), axis=1)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        test_data['positions'][:, 0], 
        test_data['positions'][:, 1],
        c=errors,
        cmap='viridis',
        s=5,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Mean Absolute Error')
    plt.title('Prediction Error Across Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('test_plots/error_heatmap.png')
    plt.close()

def plot_simulation_controls(test_data, model, params, num_simulations=4):
    """Plot simulated trajectories with different control inputs."""
    plt.figure(figsize=(15, 12))
    
    # Get a random initial state
    idx = np.random.randint(0, len(test_data['inputs']))
    initial_state = test_data['inputs'][idx].copy()
    
    # Different control scenarios
    control_scenarios = [
        {'throttle': 1.0, 'brake': 0.0, 'steer': 0.0, 'label': 'Full Throttle'},
        {'throttle': 0.5, 'brake': 0.0, 'steer': 0.3, 'label': 'Half Throttle + Right Turn'},
        {'throttle': 0.5, 'brake': 0.0, 'steer': -0.3, 'label': 'Half Throttle + Left Turn'},
        {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0, 'label': 'Full Brake'}
    ]
    
    for i, controls in enumerate(control_scenarios[:num_simulations]):
        # Create a copy of the initial state
        state = initial_state.copy()
        
        # Set control inputs
        state[7] = controls['throttle']  # Throttle
        state[8] = controls['brake']     # Brake
        state[14] = controls['steer']    # Steering
        
        # Initialize trajectory
        traj = [np.array(state[0:2].copy())]  # Convert to numpy array
        
        # Simulate for 50 steps
        for _ in range(50):
            # Prepare model input
            model_input = prepare_model_inputs(state.reshape(1, -1))
            
            # Get model prediction
            state_dot = model.apply({'params': params}, model_input)[0]
            
            # Convert JAX array to numpy if needed
            if hasattr(state_dot, 'device_buffer'):
                state_dot = np.array(state_dot)
            
            # Update state with Euler integration
            state[0:2] += state_dot[3:5] * 0.1  # Position
            state[3:5] += state_dot[3:5] * 0.1  # Velocity
            state[5] += state_dot[5] * 0.1      # Angular velocity
            state[6] += state_dot[6] * 0.1      # Heading
            
            # Add to trajectory using numpy arrays
            next_pos = np.array(traj[-1]) + np.array(state_dot[0:2]) * 0.1
            traj.append(next_pos)
        
        # Convert to numpy array for plotting
        traj = np.array(traj)
        
        # Plot
        plt.plot(traj[:, 0], traj[:, 1], label=controls['label'])
    
    plt.title('Simulated Trajectories with Different Controls')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('test_plots/simulation_controls.png')
    plt.close()

def main():
    # Create directory for plots
    Path("./test_plots").mkdir(exist_ok=True)
    
    # Load model
    model, params = load_model(Path(CONFIG['general']['save_dir']) / 'best_model.pkl')
    
    # Load test data
    test_data = load_test_data()
    
    print(f"Test data loaded with {len(test_data['inputs'])} samples")
    
    # Prepare inputs for the model
    model_inputs = prepare_model_inputs(test_data['inputs'])
    
    # Get model predictions
    predictions = model.apply({'params': params}, model_inputs)
    
    # Compute metrics
    metrics = compute_metrics(test_data['outputs'], predictions)
    
    # Print metrics
    print("\n==== Model Performance on Test Set ====")
    print(f"Overall L2 Loss: {metrics['l2_loss']:.6f}")
    print(f"L-infinity Loss: {metrics['linf_loss']:.6f}")
    print(f"Overall R2 Score: {metrics['r2_score']:.4f}")
    print(f"RMSE (overall): {metrics['rmse_overall']:.6f}")
    
    print("\nComponent-wise R2 Scores:")
    for component, r2 in metrics['component_r2'].items():
        print(f"  {component}: {r2:.4f}")
    
    print("\nComponent-wise RMSE:")
    component_names = ['pos_x', 'pos_y', 'steering', 'vel_x', 'vel_y', 'ang_vel', 'heading']
    for i, component in enumerate(component_names):
        print(f"  {component}: {metrics['rmse'][i]:.6f}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # 1. Prediction vs actual scatter plots
    plot_prediction_vs_actual(test_data['outputs'], predictions)
    
    # 2. Error distribution
    plot_error_distribution(test_data['outputs'], predictions)
    
    # 3. Trajectory comparisons
    plot_trajectory_comparison(test_data, test_data['outputs'], predictions)
    
    # 4. Error heatmap
    plot_error_heatmap(test_data, test_data['outputs'], predictions)
    
    # 5. Simulation with different controls
    plot_simulation_controls(test_data, model, params)
    
    print("Testing completed. Results saved to test_plots directory.")

def load_test_data():
    """Load the unseen test data."""
    test_data_path = Path(CONFIG['data']['test_data_path'])
    return np.load(test_data_path)

if __name__ == "__main__":
    main()