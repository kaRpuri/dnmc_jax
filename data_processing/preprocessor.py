import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import NamedTuple, Tuple

class NormalizedData(NamedTuple):
    """Container for normalized datasets"""
    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]
    scaler: object
    metadata: dict

class Preprocessor:
    """Handles normalization and temporal splitting of vehicle dynamics data"""
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (mean/std) or 'robust' (median/IQR)
        """
        self.method = method
        self.input_scaler = None
        self.output_scaler = None
        self.fitted = False

    def fit(self, inputs: np.ndarray, outputs: np.ndarray):
        """Learn normalization parameters from training data"""
        if self.method == 'standard':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        elif self.method == 'robust':
            self.input_scaler = RobustScaler()
            self.output_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.input_scaler.fit(inputs)
        self.output_scaler.fit(outputs)
        self.fitted = True

    def transform(self, inputs: np.ndarray, outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply learned normalization"""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
            
        return (
            self.input_scaler.transform(inputs),
            self.output_scaler.transform(outputs)
        )

    def inverse_transform_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Convert normalized outputs back to original scale"""
        return self.output_scaler.inverse_transform(outputs)

    def save(self, path: Path):
        """Save scaler state"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> 'Preprocessor':
        """Load scaler state"""
        with open(path, 'rb') as f:
            return pickle.load(f)

def temporal_split(
    inputs: np.ndarray,
    outputs: np.ndarray,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Split data into train/val/test sets while preserving temporal order
    
    Args:
        inputs: (N, 15) input features
        outputs: (N, 7) target derivatives
        ratios: (train, val, test) proportions summing to 1
        
    Returns:
        ((train_in, train_out), (val_in, val_out), (test_in, test_out))
    """
    assert sum(ratios) == 1.0, "Ratios must sum to 1"
    
    n = len(inputs)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    return (
        (inputs[:train_end], outputs[:train_end]),
        (inputs[train_end:val_end], outputs[train_end:val_end]),
        (inputs[val_end:], outputs[val_end:])
    )





if __name__ == "__main__":
    from data_loader import load_and_validate
    from feature_extractor import extract_features
    from preprocessor import Preprocessor, temporal_split, NormalizedData
    
    # 1. Load and process raw data
    raw = load_and_validate("../data_record_20deg.npz")
    processed = extract_features(raw)
    
    # 2. Split data
    (train_in, train_out), (val_in, val_out), (test_in, test_out) = temporal_split(
        processed.inputs, processed.targets
    )
    
    # 3. Initialize and fit preprocessor
    pre = Preprocessor(method='standard')
    pre.fit(train_in, train_out)
    
    # 4. Transform all splits
    train_in_norm, train_out_norm = pre.transform(train_in, train_out)
    val_in_norm, val_out_norm = pre.transform(val_in, val_out)
    test_in_norm, test_out_norm = pre.transform(test_in, test_out)
    
    # 5. Validate normalization
    print("\nNormalization Validation:")
    print(f"Train Input Mean: {train_in_norm.mean(axis=0)[:5]} (expected ~0)")
    print(f"Train Input Std: {train_in_norm.std(axis=0)[:5]} (expected ~1)")
    print(f"Train Output Mean: {train_out_norm.mean(axis=0)} (expected ~0)")
    
    # 6. Test inverse transform
    original_sample = train_out[0]
    normalized_sample = train_out_norm[0]
    reconstructed = pre.output_scaler.inverse_transform([normalized_sample])[0]
    error = np.abs(original_sample - reconstructed).max()
    print(f"\nInverse Transform Test Error: {error:.2e} (should be near 0)")
    
    # 7. Save processed data
    output_dir = Path("./processed")
    output_dir.mkdir(exist_ok=True)
    
    # Save datasets
    np.savez(output_dir/"train.npz", inputs=train_in_norm, outputs=train_out_norm)
    np.savez(output_dir/"val.npz", inputs=val_in_norm, outputs=val_out_norm)
    np.savez(output_dir/"test.npz", inputs=test_in_norm, outputs=test_out_norm)
    
    # Save preprocessor
    pre.save(output_dir/"preprocessor.pkl")
    
    # Save metadata
    metadata = {
        'split_ratios': (0.7, 0.15, 0.15),
        'original_samples': len(processed.inputs),
        'train_samples': len(train_in_norm),
        'val_samples': len(val_in_norm),
        'test_samples': len(test_in_norm),
        'normalization_method': 'standard'
    }
    with open(output_dir/"metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nData saved to {output_dir} directory")


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Data loading functions (included directly in script)
def load_and_validate(filepath: str):
    """Load and validate Isaac Sim data"""
    npz = np.load(filepath)
    data = dict(npz)
    
    # Required arrays
    required = ['timestamps', 'target_steering', 'root_velocity', 
                'root_pose', 'root_acceleration_base_link',
                'joint_velocity_front_left_wheel_steer',
                'joint_velocity_front_right_wheel_steer']
    
    # Validate
    for arr in required:
        if arr not in data:
            raise ValueError(f"Missing array: {arr}")
    
    return data

# Feature extraction functions (included directly in script)
def quaternion_to_yaw(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion to yaw angle (Z-axis rotation)"""
    qw, qx, qy, qz = quaternions.T
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(t3, t4)

def extract_features(raw_data):
    """Process raw data into features and targets"""
    # Input features
    inputs = np.zeros((len(raw_data['timestamps']), 15), dtype=np.float32)
    inputs[:, 0] = raw_data['target_steering'].mean(axis=1)  # Steering
    inputs[:, 1:7] = raw_data['root_velocity']  # Velocities
    inputs[:, 7] = raw_data['root_pose'][:, 6]  # qw
    inputs[:, 8:11] = raw_data['root_pose'][:, 3:6]  # qx, qy, qz
    inputs[:, 11:14] = raw_data['root_acceleration_base_link']  # Accelerations
    inputs[:, 14] = (raw_data['joint_velocity_front_left_wheel_steer'] + 
                     raw_data['joint_velocity_front_right_wheel_steer']) / 2.0
    
    # Compute derivatives
    dt = np.diff(raw_data['timestamps'])
    pos_deriv = np.diff(raw_data['root_pose'][:, :2], axis=0) / dt[:, None]
    steer_deriv = np.diff(inputs[:, 0]) / dt
    accel_deriv = raw_data['root_acceleration_base_link'][:-1, :2]
    ang_accel_z = raw_data['root_acceleration_base_link'][:-1, 2]
    yaw_deriv = np.diff(quaternion_to_yaw(inputs[:, 7:11])) / dt
    
    targets = np.column_stack([pos_deriv, steer_deriv, accel_deriv, ang_accel_z, yaw_deriv])
    
    return inputs[:-1], targets  # Align lengths

# Plotting function
def plot_vehicle_dynamics(inputs, targets, num_samples=20000):
    """Plot first N samples of vehicle dynamics"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    
    # Steering
    axs[0,0].plot(inputs[:num_samples, 0])
    axs[0,0].set_title('Steering Angle')
    
    # Velocities
    axs[0,1].plot(inputs[:num_samples, 1], label='vx')
    axs[0,1].plot(inputs[:num_samples, 2], label='vy')
    axs[0,1].set_title('Linear Velocities')
    axs[0,1].legend()
    
    # Quaternions
    axs[1,0].plot(inputs[:num_samples, 7], label='qw')
    axs[1,0].plot(inputs[:num_samples, 8], label='qx')
    axs[1,0].plot(inputs[:num_samples, 9], label='qy') 
    axs[1,0].plot(inputs[:num_samples, 10], label='qz')
    axs[1,0].set_title('Quaternion Components')
    axs[1,0].legend()
    
    # Position derivatives
    axs[1,1].plot(targets[:num_samples, 0], label='dx')
    axs[1,1].plot(targets[:num_samples, 1], label='dy')
    axs[1,1].set_title('Position Derivatives')
    axs[1,1].legend()
    
    # Yaw rate
    axs[2,0].plot(targets[:num_samples, 6])
    axs[2,0].set_title('Yaw Rate (dyaw)')
    
    axs[2,1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load your data file (update path as needed)
    data_path = "../data_record_20deg.npz"  # Update this path!
    
    try:
        # Load and process data
        raw = load_and_validate(data_path)
        inputs, targets = extract_features(raw)
        
        # Plot first 1000 samples
        plot_vehicle_dynamics(inputs, targets)
        
        print("Plots generated successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure:")
        print(f"1. The file '{data_path}' exists")
        print("2. You have numpy and matplotlib installed")
        print("3. The data file contains all required arrays")