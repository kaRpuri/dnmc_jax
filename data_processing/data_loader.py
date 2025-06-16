import numpy as np
from pathlib import Path
from typing import Dict, NamedTuple
import pprint

class RawData(NamedTuple):
    data: Dict[str, np.ndarray]
    n_samples: int
    duration: float
    sampling_rate: float
    metadata: Dict[str, object]

def load_and_validate(filepath: str) -> RawData:
    """
    Load and validate Isaac Sim .npz data.
    
    1. Checks file existence.
    2. Loads arrays into a dict.
    3. Verifies required arrays are present.
    4. Computes sample count, duration, and sampling rate.
    5. Collects array shapes and dtypes as metadata.
    """
    # 1. File check
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")  # [1]
    
    # 2. Load data
    npz = np.load(filepath)
    data = dict(npz)  # Convert to plain dict for ease of use [2]
    
    # 3. Required arrays
    required = [
        'timestamps', 'target_steering', 'root_velocity',
        'root_pose', 'root_acceleration_base_link',
        'joint_velocity_front_left_wheel_steer',
        'joint_velocity_front_right_wheel_steer'
    ]
    missing = [r for r in required if r not in data]
    if missing:
        raise ValueError(f"Missing arrays: {missing}")  # [3]
    
    # 4. Compute metadata
    ts = data['timestamps']
    n = ts.shape[0]
    duration = float(ts[-1] - ts[0])
    rate = n / duration if duration > 0 else 0.0
    
    meta = {
        'file_size_mb': file_path.stat().st_size / (1024**2),
        'array_shapes': {k: v.shape for k, v in data.items()},
        'array_dtypes': {k: str(v.dtype) for k, v in data.items()}
    }
    
    print(f"Loaded {n:,} samples over {duration:.2f}s (~{rate:.1f} Hz)")  # [2]
    
    return RawData(
        data=data,
        n_samples=n,
        duration=duration,
        sampling_rate=rate,
        metadata=meta
    )


if __name__ == "__main__":
    try:
        raw = load_and_validate("../data_record_20deg.npz")
        # Inspect metadata
        for k, v in raw.data.items():
            print(f"{k}: shape={v.shape}")
        # pprint.pprint(raw.metadata)
        # print("Metadata:", raw.metadata)
        # Sample check
        print("Timestamps sample:", raw.data['timestamps'][:5])
    except Exception as e:
        print("Error:", e)
