"""
This script fits a hybrid neural ODE (Universal Differential Equation)
to reference data from a vehicle single track drift model.

The first four states are modeled with equations from first principles, 
whereas the last three states are modeled by a neural network.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Import from local modules
from data_processing import load_mat_data, create_scalers, prepare_data, l2_loss_scaled
from models import HybridNeuralODE, create_hybrid_ode_model
from training import predict, training_loop
from visualization import plot_comparison

def main():
    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(5)
    
    # Hyperparameters
    nn_size = 5
    learning_rate = 0.025
    noise_level = 0.025
    max_iterations = 2000
    group_size = 80
    continuity_term = 1.0
    
    # Set up results directory
    results_dir = os.path.join("results", "reference_single_track_drift", "hynode")
    os.makedirs(results_dir, exist_ok=True)
    
    # Print experiment info
    print(f"\nExperiment with layer_size {nn_size}\n")
    
    # Load data
    ref_model_dir = os.path.join("..", "hybrid_node", "reference_model")
    data = load_mat_data(os.path.join(ref_model_dir, "reference_single_track_drift.mat"))
    data2 = load_mat_data(os.path.join(ref_model_dir, "reference_single_track_drift2.mat"))
    data3 = load_mat_data(os.path.join(ref_model_dir, "reference_single_track_drift3.mat"))
    
    # Create scalers
    state_scaler, input_scaler = create_scalers(data, (0.1, 69.9))
    
    # Prepare training and validation data
    vali = prepare_data(data, (70.0, 99.9), state_scaler, noise_level)
    train_datasets = [
        prepare_data(data2, (0.1, 69.9), state_scaler, noise_level),
        prepare_data(data3, (0.1, 69.9), state_scaler, noise_level),
        prepare_data(data, (0.1, 69.9), state_scaler, noise_level)
    ]
    
    # Initialize neural network
    model = HybridNeuralODE(hidden_size=nn_size)
    rng_key, init_key = jax.random.split(rng_key)
    
    # Create sample input for initialization
    sample_input = jnp.ones((6,))
    params = model.init(init_key, sample_input)
    
    # Create hybrid ODE models
    hybrid_models = []
    for dataset in train_datasets:
        hybrid_model = create_hybrid_ode_model(
            model.apply, params, state_scaler, input_scaler, dataset.input_interpolant
        )
        hybrid_models.append(hybrid_model)
    
    # Training
    trained_params = training_loop(
        hybrid_models[0],  # Start with the first model
        params,
        train_datasets,
        state_scaler,
        input_scaler,
        learning_rate=learning_rate,
        max_iter=max_iterations,
        group_size=group_size,
        continuity_term=continuity_term
    )
    
    # Create final model for last training dataset
    final_train = train_datasets[-1]
    final_model = create_hybrid_ode_model(
        model.apply, trained_params, state_scaler, input_scaler, final_train.input_interpolant
    )
    
    # Predict on training data
    train_predictions = predict(
        lambda y, t, args: final_model(y, t, trained_params),
        trained_params,
        final_train.states[:, 0],
        final_train.tspan,
        final_train.tsteps
    )
    
    # Calculate training error
    train_err = l2_loss_scaled(final_train.states, train_predictions, state_scaler)
    print(f"Training error: {train_err}")
    
    # Create validation model
    vali_model = create_hybrid_ode_model(
        model.apply, trained_params, state_scaler, input_scaler, vali.input_interpolant
    )
    
    # Predict on validation data
    vali_predictions = predict(
        lambda y, t, args: vali_model(y, t, trained_params),
        trained_params,
        vali.states[:, 0],
        vali.tspan,
        vali.tsteps
    )
    
    # Calculate validation error
    vali_err = l2_loss_scaled(vali.states, vali_predictions, state_scaler)
    print(f"Validation error: {vali_err}")
    
    # Save results
    results = pd.DataFrame({
        "train_err": [train_err],
        "pred_err": [vali_err]
    })
    results.to_csv(os.path.join(results_dir, f"hynode_{nn_size}.csv"), index=False)
    
    # Combine data for plotting
    all_tsteps = np.concatenate([final_train.tsteps, vali.tsteps])
    all_states = np.hstack([final_train.states, vali.states])
    all_predictions = np.hstack([train_predictions, vali_predictions])
    
    # Plot comparison
    plot_comparison(
        all_tsteps,
        all_states,
        all_predictions,
        vline_pos=70.0,  # Mark the boundary between train and validation
        save_path=os.path.join(results_dir, f"hynode_val_{nn_size}.png")
    )
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()