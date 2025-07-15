import jax
import jax.numpy as jnp
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any, Callable

class HybridNeuralODE(nn.Module):
    """Neural network component of the hybrid model"""
    hidden_size: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = jnp.tanh(x)
        x = nn.Dense(3)(x)
        return x

def create_hybrid_ode_model(nn_apply, nn_params, state_scaler, input_scaler, input_interpolant):
    """Create hybrid ODE function combining physics and neural network"""
    
    def ude_dynamics(state, t, params):
        """Combined physics + neural network dynamics"""
        # Extract current state values
        x, y, psi, delta, v, beta, psi_dot = state
        
        # Get control inputs at time t
        ax = input_interpolant[0](t)
        v_delta = input_interpolant[1](t)
        
        # Physics-based equations for first 4 states
        dx = v * jnp.cos(psi + beta)
        dy = v * jnp.sin(psi + beta)
        dpsi = psi_dot
        ddelta = v_delta
        
        # Scale states and inputs for neural network
        state_scaled = state_scaler.transform(state.reshape(1, -1))[0]
        input_scaled = input_scaler.transform(jnp.array([[ax, v_delta]]))[0]
        
        # Neural network takes steering angle, velocity, side slip, yaw rate + control inputs
        nn_input = jnp.concatenate([state_scaled[3:7], input_scaled])
        
        # Neural network predicts derivatives of v, beta, and psi_dot
        dv_dbeta_dpsi_dot = nn_apply(params, nn_input)
        
        # Combine physics and NN outputs
        derivatives = jnp.array([
            dx, dy, dpsi, ddelta, 
            dv_dbeta_dpsi_dot[0], 
            dv_dbeta_dpsi_dot[1], 
            dv_dbeta_dpsi_dot[2]
        ])
        
        return derivatives
    
    return ude_dynamics