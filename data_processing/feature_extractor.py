import numpy as np
from typing import NamedTuple

class ProcessedData(NamedTuple):
    """Container for processed features and targets"""
    inputs: np.ndarray  # (N, 15) network input features
    targets: np.ndarray  # (N-1, 7) delta states (derivatives)
    timestamps: np.ndarray  # (N-1,) aligned timestamps

def extract_features(raw_data, jump_thr: float = 0.002) -> ProcessedData:
    """
    Converts raw Isaac Sim data to model-ready features and targets.
    Removes samples with abrupt steering transitions internally.
    """
    # 1. Remove abrupt steering transitions
    steering = raw_data.data['target_steering'].mean(axis=1)
    cleaned_data, _ = remove_steering_jumps(steering, raw_data.data, jump_thr=jump_thr, buffer=1)
    n_samples = len(cleaned_data['timestamps'])

    # 2. Input Feature Extraction
    inputs = np.zeros((n_samples, 15), dtype=np.float32)
    inputs[:, 0] = cleaned_data['target_steering'].mean(axis=1)
    inputs[:, 1:7] = cleaned_data['root_velocity']
    inputs[:, 7] = cleaned_data['root_pose'][:, 6]  # qw
    inputs[:, 8:11] = cleaned_data['root_pose'][:, 3:6]  # qx, qy, qz
    inputs[:, 11:14] = cleaned_data['root_acceleration_base_link']
    steer_vel_left = cleaned_data['joint_velocity_front_left_wheel_steer']
    steer_vel_right = cleaned_data['joint_velocity_front_right_wheel_steer']
    inputs[:, 14] = (steer_vel_left + steer_vel_right) / 2.0

    # 3. Delta State Computation
    dt = np.diff(cleaned_data['timestamps'])
    pos = cleaned_data['root_pose'][:, :2]
    pos_deriv = np.diff(pos, axis=0) / dt[:, None]
    steering = inputs[:, 0]
    steer_deriv = np.diff(steering) / dt
    accel_deriv = cleaned_data['root_acceleration_base_link'][:-1, :2]
    ang_accel_z = cleaned_data['root_acceleration_base_link'][:-1, 2]
    yaw = quaternion_to_yaw(inputs[:, 7:11])
    yaw_deriv = np.diff(yaw) / dt

    targets = np.column_stack([
        pos_deriv,        # dx, dy
        steer_deriv,      # d_steering
        accel_deriv,      # dvx, dvy
        ang_accel_z,      # dwz
        yaw_deriv         # dyaw
    ])
    inputs = inputs[:-1]
    timestamps = cleaned_data['timestamps'][:-1]
    return ProcessedData(inputs, targets, timestamps)
def remove_steering_jumps(steering: np.ndarray, data_arrays: dict, jump_thr: float = 0.002, buffer: int = 1):
    """
    Remove samples where the change in steering angle exceeds jump_thr.
    
    Args:
        steering: 1D numpy array of steering angles (shape: N,)
        data_arrays: Dictionary of arrays (all shape N or (N, ...)) to be filtered.
        jump_thr: Threshold for detecting abrupt steering changes.
        buffer: Number of samples before and after each jump to remove.
    
    Returns:
        filtered_arrays: Dictionary of arrays with abrupt transitions removed.
        kept_mask: Boolean mask indicating which samples are kept.
    """
    # Find indices where steering changes abruptly
    diffs = np.abs(np.diff(steering))
    boundaries = np.where(diffs > jump_thr)[0]
    
    # Build mask: start with all True
    mask = np.ones_like(steering, dtype=bool)
    for idx in boundaries:
        for offset in range(-buffer, buffer + 1):
            j = idx + offset
            if 0 <= j < len(mask):
                mask[j] = False
    
    # Apply mask to all arrays
    filtered_arrays = {k: v[mask] for k, v in data_arrays.items()}
    return filtered_arrays, mask


def quaternion_to_yaw(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion array to yaw angles (radians)"""
    qw, qx, qy, qz = quaternions.T
    
    # Normalize quaternions
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm
    
    # Yaw calculation
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(t3, t4)


if __name__ == "__main__":
    from data_loader import load_and_validate
    
    # Load test data
    raw = load_and_validate("../data_record_20deg.npz")
    
    # Process features
    processed = extract_features(raw)
    
    print("\nFeature Engineering Validation:")
    print(f"Inputs shape: {processed.inputs.shape} (expect (124999, 15))")
    print(f"Targets shape: {processed.targets.shape} (expect (124999, 7))")
    print(f"Timestamp shape: {processed.timestamps.shape}")
    
    # Verify first sample
    print("\nFirst input sample:", processed.inputs[0])
    print("First target sample:", processed.targets[0])
    import matplotlib.pyplot as plt
    import numpy as np

    # Use your actual processed data here:
    # inputs = processed.inputs
    # targets = processed.targets

    # For demonstration, here's how to select the first 1000 samples:
    N = 125000
    inputs_plot = processed.inputs[:N]  # Access .inputs from ProcessedData
    targets_plot = processed.targets[:N]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Steering anglef
    axs[0, 0].plot(inputs_plot[:, 0])
    axs[0, 0].set_title('Steering Angle over Time')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Steering Angle (rad)')

    # Vehicle linear velocities vx, vy
    axs[0, 1].plot(inputs_plot[:, 1], label='vx')
    axs[0, 1].plot(inputs_plot[:, 2], label='vy')
    axs[0, 1].set_title('Vehicle Linear Velocities')
    axs[0, 1].set_xlabel('Sample')
    axs[0, 1].set_ylabel('Velocity (m/s)')
    axs[0, 1].legend()

    # Quaternion components
    axs[1, 0].plot(inputs_plot[:, 7], label='qw')
    axs[1, 0].plot(inputs_plot[:, 8], label='qx')
    axs[1, 0].plot(inputs_plot[:, 9], label='qy')
    axs[1, 0].plot(inputs_plot[:, 10], label='qz')
    axs[1, 0].set_title('Quaternion Components')
    axs[1, 0].set_xlabel('Sample')
    axs[1, 0].set_ylabel('Quaternion Value')
    axs[1, 0].legend()

    # Position derivatives dx, dy
    axs[1, 1].plot(targets_plot[:, 0], label='dx')
    axs[1, 1].plot(targets_plot[:, 1], label='dy')
    axs[1, 1].set_title('Position Derivatives')
    axs[1, 1].set_xlabel('Sample')
    axs[1, 1].set_ylabel('Derivative (m/s)')
    axs[1, 1].legend()

    # Yaw derivative dyaw
    axs[2, 0].plot(targets_plot[:, 6])
    axs[2, 0].set_title('Yaw Derivative (dyaw)')
    axs[2, 0].set_xlabel('Sample')
    axs[2, 0].set_ylabel('Yaw Rate (rad/s)')

    # Hide unused subplot
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    max_hist = min(100000, processed.inputs.shape[0], processed.targets.shape[0])
    inputs_hist = processed.inputs[:max_hist]
    targets_hist = processed.targets[:max_hist]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Histograms of Control Inputs and State Features', fontsize=16)

    # Row 1: Steering angle, steering speed, vx
    axs[0, 0].hist(inputs_hist[:, 0], bins=50, color='blue', alpha=0.7)
    axs[0, 0].set_title('Steering Angle')
    axs[0, 0].set_xlabel('Steering Angle (rad)')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(inputs_hist[:, 14], bins=50, color='green', alpha=0.7)
    axs[0, 1].set_title('Steering Speed')
    axs[0, 1].set_xlabel('Steering Speed (rad/s)')
    axs[0, 1].set_ylabel('Frequency')

    axs[0, 2].hist(inputs_hist[:, 1], bins=50, color='red', alpha=0.7)
    axs[0, 2].set_title('Linear Velocity vx')
    axs[0, 2].set_xlabel('vx (m/s)')
    axs[0, 2].set_ylabel('Frequency')

    # Row 2: vy, vz, dx
    axs[1, 0].hist(inputs_hist[:, 2], bins=50, color='orange', alpha=0.7)
    axs[1, 0].set_title('Linear Velocity vy')
    axs[1, 0].set_xlabel('vy (m/s)')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].hist(inputs_hist[:, 3], bins=50, color='purple', alpha=0.7)
    axs[1, 1].set_title('Linear Velocity vz')
    axs[1, 1].set_xlabel('vz (m/s)')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 2].hist(targets_hist[:, 0], bins=50, color='cyan', alpha=0.7)
    axs[1, 2].set_title('Position Derivative dx')
    axs[1, 2].set_xlabel('dx (m/s)')
    axs[1, 2].set_ylabel('Frequency')

    # Row 3: dy, wz, dyaw
    axs[2, 0].hist(targets_hist[:, 1], bins=50, color='magenta', alpha=0.7)
    axs[2, 0].set_title('Position Derivative dy')
    axs[2, 0].set_xlabel('dy (m/s)')
    axs[2, 0].set_ylabel('Frequency')

    axs[2, 1].hist(inputs_hist[:, 6], bins=50, color='brown', alpha=0.7)
    axs[2, 1].set_title('Angular Velocity wz')
    axs[2, 1].set_xlabel('wz (rad/s)')
    axs[2, 1].set_ylabel('Frequency')

    axs[2, 2].hist(targets_hist[:, 6], bins=50, color='black', alpha=0.7)
    axs[2, 2].set_title('Yaw Derivative dyaw')
    axs[2, 2].set_xlabel('dyaw (rad/s)')
    axs[2, 2].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

            