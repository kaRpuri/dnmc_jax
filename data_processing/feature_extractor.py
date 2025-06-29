import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import NamedTuple, List
import os
import sys


class ProcessedData(NamedTuple):
    inputs: np.ndarray    # (N, 15)
    targets: np.ndarray   # (N, 7)
    timestamps: np.ndarray  # (N,)
    positions: np.ndarray   # (N, 2)

def load_npz_file(npz_path):
    if not os.path.isfile(npz_path):
        print(f"Error: file not found: {npz_path}", file=sys.stderr)
        return
    # Load the archive (this gives you a dict‐like object)
    data = np.load(npz_path, allow_pickle=True)
    keys = data.files

    if not keys:
        print("No arrays found in the .npz file.")
        return

    # print the keys / data
    print(f"Found {len(keys)} array(s) in '{npz_path}':")
    for key in keys:
        arr = data[key]
        print(f"  --> {key}  (shape={arr.shape}, dtype={arr.dtype})")

    return data

def extract_features(raw_data):
    """Create inputs and targets from raw data, grouped by robot ID"""
    robot_ids = raw_data['robot_id'].flatten()
    unique_ids = np.unique(robot_ids)
    
    all_inputs = []
    all_targets = []
    all_timestamps = []
    all_positions = []
    
    for rid in unique_ids:
        mask = (robot_ids == rid)
        ts = raw_data['timestamps'][mask].flatten()
        positions = raw_data['root_pose'][mask, :2]
        steering = raw_data['target_steering'][mask].mean(axis=1)
        
        # Assemble inputs for this robot (15 dimensions)
        X = np.empty((len(ts), 15), np.float32)
        X[:, 0:6] = raw_data['root_velocity'][mask]
        q = raw_data['root_pose'][mask, 3:7]
        X[:, 6:10] = q[:, [3, 0, 1, 2]]  # qw, qx, qy, qz
        X[:, 10:13] = raw_data['root_acceleration_base_link'][mask]
        X[:, 13] = raw_data['target_velocity'][mask, 0]
        X[:, 14] = steering
        
        # Compute derivatives (targets) for this robot
        dt = np.diff(ts)
        if len(dt) == 0:  # Skip if only one data point
            continue
            
        dxdy = np.diff(positions, axis=0) / dt[:, None]
        dsteer = np.diff(X[:, 14]) / dt
        dv = np.diff(X[:, 0:2], axis=0) / dt[:, None]
        wz = raw_data['root_velocity'][mask, 5]
        dwz = np.diff(wz) / dt
        
        # Use angular velocity for yaw derivative
        dyaw = wz[:-1]  # Current angular velocity as yaw derivative
        
        Y = np.column_stack([dxdy, dsteer, dv, dwz, dyaw])
        
        # Only keep states where we have next state (N-1)
        all_inputs.append(X[:-1])
        all_targets.append(Y)
        all_timestamps.append(ts[:-1])
        all_positions.append(positions[:-1])
    
    # Combine all robots' data
    return ProcessedData(
        inputs=np.vstack(all_inputs),
        targets=np.vstack(all_targets),
        timestamps=np.concatenate(all_timestamps),
        positions=np.vstack(all_positions)
    )

def filter_invalid_transitions(processed, error_threshold=0.005):
    """Filter out data points with integration error > threshold"""
    print("\nFiltering invalid transitions:")
    labels = ["dx", "dy", "dsteer", "dvx", "dvy", "dwz", "dyaw"]
    n = min(len(processed.targets), len(processed.timestamps) - 1)
    valid_mask = np.ones(n, dtype=bool)
    all_predicted = []
    all_actual = []
    count = 0
    
    for i in range(n):
        dt = processed.timestamps[i+1] - processed.timestamps[i]
        
        # Current state
        current_state = processed.inputs[i]
        current_pos = processed.positions[i]
        current_steer = current_state[14]
        
        # Compute predicted next state
        integrated_pos = current_pos + processed.targets[i, 0:2] * dt
        integrated_steer = current_steer + processed.targets[i, 2] * dt
        integrated_vx = current_state[0] + processed.targets[i, 3] * dt
        integrated_vy = current_state[1] + processed.targets[i, 4] * dt
        integrated_wz = current_state[5] + processed.targets[i, 5] * dt
        integrated_yaw = Rotation.from_quat(current_state[6:10]).as_euler('zyx')[0] + processed.targets[i, 6] * dt
        
        # Actual next state
        next_state = processed.inputs[i+1]
        actual_pos = processed.positions[i+1]
        actual_steer = next_state[14]
        actual_vx, actual_vy, actual_wz = next_state[0], next_state[1], next_state[5]
        actual_yaw = Rotation.from_quat(next_state[6:10]).as_euler('zyx')[0]
        
        # Store for plotting
        predicted_state = [integrated_pos[0], integrated_pos[1], integrated_steer,
                          integrated_vx, integrated_vy, integrated_wz, integrated_yaw]
        actual_state = [actual_pos[0], actual_pos[1], actual_steer,
                       actual_vx, actual_vy, actual_wz, actual_yaw]
        
        all_predicted.append(predicted_state)
        all_actual.append(actual_state)
        
        # Calculate errors
        errors = [
            abs(integrated_pos[0] - actual_pos[0]),
            abs(integrated_pos[1] - actual_pos[1]),
            abs(integrated_steer - actual_steer),
            abs(integrated_vx - actual_vx),
            abs(integrated_vy - actual_vy),
            abs(integrated_wz - actual_wz),
            abs(integrated_yaw - actual_yaw)
        ]
        
        # Check if any error exceeds threshold
        if any(err > error_threshold for err in errors):
            valid_mask[i] = False
            count += 1
    
    print(f"Removed {count} data points with errors > {error_threshold}")
    
    # Create filtered dataset
    current_inputs = processed.inputs[:n]
    current_positions = processed.positions[:n]
    current_timestamps = processed.timestamps[:n]
    
    filtered_inputs = current_inputs[valid_mask]
    filtered_targets = processed.targets[:n][valid_mask]
    filtered_timestamps = current_timestamps[valid_mask]
    filtered_positions = current_positions[valid_mask]
    
    # Convert to arrays for plotting
    predicted_array = np.array(all_predicted)
    actual_array = np.array(all_actual)
    
    # Filter prediction arrays
    filtered_predicted = predicted_array[valid_mask]
    filtered_actual = actual_array[valid_mask]
    
    return ProcessedData(
        filtered_inputs,
        filtered_targets,
        filtered_timestamps,
        filtered_positions, # Steering angles from filtered inputs
    ), filtered_predicted, filtered_actual



def plot_delta_comparison(predicted, actual, labels):
    """Plot integrated vs actual for all delta states"""
    plt.figure(figsize=(15, 10))
    for i in range(7):
        plt.subplot(4, 2, i+1)
        plt.plot(predicted[:, i], 'b-', label='Integrated')
        plt.plot(actual[:, i], 'r--', label='Actual')
        plt.title(f'State: {labels[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.savefig('delta_state_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Load and process data
    raw_data = load_npz_file("../data_record_20deg.npz")
    processed = extract_features(raw_data)
    filtered, predicted, actual  = filter_invalid_transitions(processed)
    # filtered = remove_steering_jumps(processed)
    
    # Print sample data
    print("\n=== First 2 Input Samples ===")
    input_labels = ["vx","vy","vz","wx","wy","wz","qw","qx","qy","qz","ax","ay","az","v_cmd","steer_cmd"]
    for i in range(2):
        print(f"\nSample {i}:")
        for j, label in enumerate(input_labels):
            print(f"  {label}: {filtered.inputs[i, j]:.6f}")
    
    print("\n=== First Target Sample ===")
    target_labels = ["dx","dy","dsteer","dvx","dvy","dwz","dyaw"]
    for j, label in enumerate(target_labels):
        print(f"  {label}: {filtered.targets[0, j]:.6f}")
    
    # Validate integration

    
    print(f"\nNumber of input samples: {len(filtered.inputs)}")
    print(f"Number of target samples: {len(filtered.targets)}")
    
    
    
    
    # Plot comparison of all delta states
    plot_delta_comparison(predicted, actual, ["Pos X", "Pos Y", "Steering", "Vel X", "Vel Y", "Ang Vel Z", "Yaw"])
    
    # Plot histograms
    plt.figure(figsize=(12, 8))
    features = {
        'Velocity X': filtered.inputs[:, 0],
        'Velocity Y': filtered.inputs[:, 1],
        'Commanded Velocity': filtered.inputs[:, 13],
        'Commanded Steering': filtered.inputs[:, 14]
    }

    for i, (title, data) in enumerate(features.items()):
        plt.subplot(2, 2, i+1)
        plt.hist(data, bins=50, alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('feature_histograms.png')
    plt.show()
    
    # Plot trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(filtered.positions[:, 0], filtered.positions[:, 1], 'b-', label='Actual Path')
    plt.title("Vehicle Trajectory")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('vehicle_trajectory.png')
    plt.show()


