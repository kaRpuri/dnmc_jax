import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict, Any

@dataclass
class SimulationData:
    """Container for simulation data, equivalent to Julia's Sim struct"""
    tspan: Tuple[float, float]
    tsteps: np.ndarray
    states: np.ndarray
    scaled_states: np.ndarray
    states_interpolant: List[Callable]
    input_interpolant: List[Callable]

def load_mat_data(file_path: str) -> Dict[str, Any]:
    """Load data from MAT file"""
    return sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

def create_state_interpolants(data: Dict) -> List[Callable]:
    """Create interpolation functions for states"""
    t = data["t_Ref"]
    
    # Create interpolant for each state
    interpolants = [
        interp1d(t, data["x_State"].xPos),
        interp1d(t, data["x_State"].yPos),
        interp1d(t, data["x_State"].psi),
        interp1d(t, data["x_State"].delta),
        interp1d(t, data["x_State"].v),
        interp1d(t, data["x_State"].beta),
        interp1d(t, data["x_State"].dPsi)
    ]
    
    return interpolants

def create_input_interpolants(data: Dict) -> List[Callable]:
    """Create interpolation functions for inputs"""
    t = data["t_Ref"]
    
    # Create interpolant for each input
    interpolants = [
        interp1d(t, data["u_Input"].U_ax),
        interp1d(t, data["u_Input"].U_vdelta)
    ]
    
    return interpolants

def interpolate_vector(interpolants: List[Callable], t: float) -> np.ndarray:
    """Apply interpolation functions at time t"""
    return np.array([func(t) for func in interpolants])

def interpolate_matrix(interpolants: List[Callable], t_array: np.ndarray) -> np.ndarray:
    """Interpolate values for multiple time points"""
    result = np.zeros((len(interpolants), len(t_array)))
    for i, t in enumerate(t_array):
        result[:, i] = interpolate_vector(interpolants, t)
    return result

def create_scalers(data: Dict, tspan: Tuple[float, float]) -> Tuple[StandardScaler, StandardScaler]:
    """Create scalers for states and inputs"""
    # Create time steps
    tsteps = np.arange(tspan[0], tspan[1] + 0.1, 0.1)
    
    # Create interpolants
    states_interpolant = create_state_interpolants(data)
    input_interpolant = create_input_interpolants(data)
    
    # Get matrices
    states = interpolate_matrix(states_interpolant, tsteps)
    inputs = interpolate_matrix(input_interpolant, tsteps)
    
    # Create and fit scalers
    state_scaler = StandardScaler().fit(states.T)
    input_scaler = StandardScaler().fit(inputs.T)
    
    return state_scaler, input_scaler

def prepare_data(data: Dict, tspan: Tuple[float, float], 
                state_scaler: StandardScaler, noise_level: float = 0.025) -> SimulationData:
    """Prepare data for training or validation"""
    # Create time steps
    tsteps = np.arange(tspan[0], tspan[1] + 0.1, 0.1)
    
    # Create interpolants
    states_interpolant = create_state_interpolants(data)
    input_interpolant = create_input_interpolants(data)
    
    # Get states matrix
    states = interpolate_matrix(states_interpolant, tsteps)
    
    # Scale states
    scaled_states = state_scaler.transform(states.T).T
    
    # Add noise
    scaled_states += np.random.normal(0, noise_level, scaled_states.shape)
    
    # Reconstruct states with noise
    states = state_scaler.inverse_transform(scaled_states.T).T
    
    return SimulationData(
        tspan=tspan,
        tsteps=tsteps,
        states=states,
        scaled_states=scaled_states,
        states_interpolant=states_interpolant,
        input_interpolant=input_interpolant
    )

def l2_loss(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate L2 loss between target and prediction"""
    return np.sum((target - prediction) ** 2)

def l2_loss_scaled(target: np.ndarray, prediction: np.ndarray, scaler: StandardScaler) -> float:
    """Calculate L2 loss in scaled space"""
    target_scaled = scaler.transform(target.T).T
    prediction_scaled = scaler.transform(prediction.T).T
    return l2_loss(target_scaled, prediction_scaled)