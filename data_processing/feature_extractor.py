import numpy as np
from typing import NamedTuple

class ProcessedData(NamedTuple):
    """Container for processed features and targets"""
    inputs: np.ndarray  # (N, 15) network input features
    targets: np.ndarray  # (N-1, 7) delta states (derivatives)
    timestamps: np.ndarray  # (N-1,) aligned timestamps

def extract_features(raw_data) -> ProcessedData:
    """
    Converts raw Isaac Sim data to model-ready features and targets.
    
    Steps:
    1. Extract 15D input features from raw arrays
    2. Compute 7D delta states via numerical differentiation
    3. Align timestamps with derivative calculations
    """
    # 1. Input Feature Extraction 
    n_samples = raw_data.n_samples
    
    # Initialize input array (N, 15)
    inputs = np.zeros((n_samples, 15), dtype=np.float32)
    
    # Steering (average of front wheels)
    inputs[:, 0] = raw_data.data['target_steering'].mean(axis=1)
    
    # Velocities [vx, vy, vz, wx, wy, wz]
    inputs[:, 1:7] = raw_data.data['root_velocity']
    
    # Quaternion [qw, qx, qy, qz] from root_pose
    inputs[:, 7] = raw_data.data['root_pose'][:, 6]  # qw
    inputs[:, 8:11] = raw_data.data['root_pose'][:, 3:6]  # qx, qy, qz
    
    # Accelerations [ax, ay, az]
    inputs[:, 11:14] = raw_data.data['root_acceleration_base_link']
    
    # Steering speed (average of front wheel steering velocities)
    steer_vel_left = raw_data.data['joint_velocity_front_left_wheel_steer']
    steer_vel_right = raw_data.data['joint_velocity_front_right_wheel_steer']
    inputs[:, 14] = (steer_vel_left + steer_vel_right) / 2.0

    # 2. Delta State Computation
    dt = np.diff(raw_data.data['timestamps'])
    
    # Position derivatives (dx, dy)
    pos = raw_data.data['root_pose'][:, :2]  # x,y only
    pos_deriv = np.diff(pos, axis=0) / dt[:, None]
    
    # Steering derivative
    steering = inputs[:, 0]
    steer_deriv = np.diff(steering) / dt
    
    # Acceleration derivatives (direct from data)
    accel_deriv = raw_data.data['root_acceleration_base_link'][:-1, :2]  # dvx, dvy
    ang_accel_z = raw_data.data['root_acceleration_base_link'][:-1, 2]  # dwz
    
    # Yaw derivative from quaternion
    yaw = quaternion_to_yaw(inputs[:, 7:11])
    yaw_deriv = np.diff(yaw) / dt
    
    # Combine targets (N-1, 7)
    targets = np.column_stack([
        pos_deriv,        # dx, dy
        steer_deriv,      # d_steering
        accel_deriv,      # dvx, dvy
        ang_accel_z,      # dwz
        yaw_deriv         # dyaw
    ])
    
    # 3. Temporal Alignment 
    inputs = inputs[:-1]
    timestamps = raw_data.data['timestamps'][:-1]
    
    return ProcessedData(inputs, targets, timestamps)

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
    N = 20000
    inputs_plot = processed.inputs[:N]  # Access .inputs from ProcessedData
    targets_plot = processed.targets[:N]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Steering angle
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

        