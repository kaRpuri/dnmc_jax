import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optax
from typing import Tuple, List, Dict, Any, Callable
from sklearn.preprocessing import StandardScaler
from .data_processing import l2_loss_scaled, SimulationData

def predict(dynamics_fn, params, initial_state, tspan, tsteps, 
            solver=diffrax.Tsit5(), rtol=1e-6, atol=1e-8):
    """Solve ODE with given parameters and return predictions at specified times"""
    solution = diffrax.diffeqsolve(
        diffrax.ODEProblem(dynamics_fn, None),
        solver,
        t0=tspan[0],
        t1=tspan[1],
        dt0=0.01,
        y0=initial_state,
        saveat=diffrax.SaveAt(ts=tsteps),
        atol=atol,
        rtol=rtol
    )
    
    return np.array(solution.ys)

def multiple_shoot(params, dynamics_fn, states, tsteps, initial_state, 
                  state_scaler, group_size=80, continuity_term=1.0):
    """Multiple shooting optimization approach"""
    n_states = states.shape[0]
    n_steps = states.shape[1]
    
    # Split trajectory into segments
    n_segments = max(1, n_steps // group_size)
    segment_indices = np.array_split(np.arange(n_steps), n_segments)
    
    total_loss = 0.0
    previous_predictions = None
    
    for i, indices in enumerate(segment_indices):
        segment_times = tsteps[indices]
        segment_states = states[:, indices]
        
        # Solve ODE for this segment
        if i == 0:
            segment_initial = initial_state
        else:
            segment_initial = previous_predictions[-1]  # Final state from previous segment
            
        segment_tspan = (segment_times[0], segment_times[-1])
        
        predictions = predict(
            lambda y, t, args: dynamics_fn(y, t, params),
            params, 
            segment_initial, 
            segment_tspan, 
            segment_times
        )
        
        # Data fitting loss
        fit_loss = l2_loss_scaled(segment_states, predictions, state_scaler)
        total_loss += fit_loss
        
        # Continuity constraint between segments
        if i > 0:
            prev_end = previous_predictions[-1]
            curr_start = predictions[0]
            continuity_loss = jnp.mean((prev_end - curr_start) ** 2)
            total_loss += continuity_term * continuity_loss
            
        previous_predictions = predictions
        
    return total_loss

def train_model(nn_model, nn_params, train_data: SimulationData, state_scaler, input_scaler, 
               previous_params=None, learning_rate=0.025, max_iter=2000, 
               group_size=80, continuity_term=1.0):
    """Train the hybrid neural ODE on a single dataset"""
    params = previous_params if previous_params is not None else nn_params
    
    # Create hybrid ODE model
    dynamics_fn = lambda state, t, params: nn_model(
        state, t, params, train_data.input_interpolant)
    
    # Define loss function
    @jax.jit
    def loss_fn(params):
        return multiple_shoot(
            params, 
            dynamics_fn,
            train_data.states,
            train_data.tsteps,
            train_data.states[:, 0],
            state_scaler,
            group_size=group_size,
            continuity_term=continuity_term
        )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    losses = []
    
    for i in range(max_iter):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        losses.append(loss_val)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss_val}")
    
    return params, losses

def training_loop(nn_model, nn_params, train_datasets, state_scaler, input_scaler,
                 learning_rate=0.025, max_iter=2000, 
                 group_size=80, continuity_term=1.0):
    """Full training loop across multiple datasets"""
    current_params = nn_params
    
    for i, dataset in enumerate(train_datasets):
        print(f"\nTraining on dataset {i+1}/{len(train_datasets)}")
        current_params, _ = train_model(
            nn_model, 
            current_params, 
            dataset, 
            state_scaler, 
            input_scaler,
            learning_rate=learning_rate,
            max_iter=max_iter,
            group_size=group_size,
            continuity_term=continuity_term
        )
    
    return current_params