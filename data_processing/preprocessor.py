import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import NamedTuple, Tuple
import yaml

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
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Split data into train/val/test sets while preserving temporal order
    
    Args:
        inputs: (N, 15) input features
        outputs: (N, 7) target derivatives
        train_ratio: proportion of data to use for training
        val_ratio: proportion of data to use for validation
        
    Returns:
        ((train_in, train_out), (val_in, val_out), (test_in, test_out))
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, \
        "Train and validation ratios must be between 0 and 1, and their sum must be less than 1"
    
    n = len(inputs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return (
        (inputs[:train_end], outputs[:train_end]),
        (inputs[train_end:val_end], outputs[train_end:val_end]),
        (inputs[val_end:], outputs[val_end:])
    )

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

def main():
    from feature_extractor import load_npz_file, extract_features, filter_invalid_transitions
    
    # 1. Load and process raw data
    raw_data_path = CONFIG['data']['raw_data_path']
    control_data_path = CONFIG['data']['control_data_path']
    
    raw = load_npz_file(raw_data_path)
    processed = extract_features(raw, control_data_path)
    
    # 1.5 Filter invalid transitions
    filtered_data, predicted, actual = filter_invalid_transitions(processed)
    
    # 2. Shuffle filtered data
    rng = np.random.default_rng(seed=CONFIG['general']['seed'])
    indices = rng.permutation(len(filtered_data.inputs))
    shuffled_inputs = filtered_data.inputs[indices]
    shuffled_targets = filtered_data.targets[indices]
    shuffled_timestamps = filtered_data.timestamps[indices]
    shuffled_positions = filtered_data.positions[indices]
    
    # 3. Split filtered data into train/val/test
    train_ratio = CONFIG['data']['split_ratio'] if 'split_ratio' in CONFIG['data'] else 0.7
    val_ratio = CONFIG['data']['val_ratio'] if 'val_ratio' in CONFIG['data'] else 0.2
    
    (train_in, train_out), (val_in, val_out), (test_in, test_out) = temporal_split(
        shuffled_inputs, shuffled_targets,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    # 4. Initialize and fit preprocessor
    pre = Preprocessor(method=CONFIG['data']['normalization_method'] if 'normalization_method' in CONFIG['data'] else 'standard')
    pre.fit(train_in, train_out)
    
    # 5. Transform all splits
    train_in_norm, train_out_norm = pre.transform(train_in, train_out)
    val_in_norm, val_out_norm = pre.transform(val_in, val_out)
    test_in_norm, test_out_norm = pre.transform(test_in, test_out)
    
    # 6. Validate normalization
    print("\nNormalization Validation:")
    print(f"Train Input Mean: {train_in_norm.mean(axis=0)[:5]} (expected ~0)")
    print(f"Train Input Std: {train_in_norm.std(axis=0)[:5]} (expected ~1)")
    print(f"Train Output Mean: {train_out_norm.mean(axis=0)} (expected ~0)")
    
    # 7. Test inverse transform
    original_sample = train_out[0]
    normalized_sample = train_out_norm[0]
    reconstructed = pre.output_scaler.inverse_transform([normalized_sample])[0]
    error = np.abs(original_sample - reconstructed).max()
    print(f"\nInverse Transform Test Error: {error:.2e} (should be near 0)")
    
    # 8. Save processed data
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Save datasets with positions and timestamps
    np.savez(processed_dir / "train.npz", 
             inputs=train_in_norm, 
             outputs=train_out_norm,
             positions=shuffled_positions[:len(train_in)],
             timestamps=shuffled_timestamps[:len(train_in)])
    
    np.savez(processed_dir / "val.npz",
             inputs=val_in_norm,
             outputs=val_out_norm,
             positions=shuffled_positions[len(train_in):len(train_in)+len(val_in)],
             timestamps=shuffled_timestamps[len(train_in):len(train_in)+len(val_in)])
    
    np.savez(processed_dir / "test.npz",
             inputs=test_in_norm,
             outputs=test_out_norm,
             positions=shuffled_positions[len(train_in)+len(val_in):],
             timestamps=shuffled_timestamps[len(train_in)+len(val_in):])
    
    # Save metadata
    metadata = {
        'split_ratio': train_ratio,
        'val_ratio': val_ratio,
        'original_samples': len(processed.inputs),
        'train_samples': len(train_in_norm),
        'val_samples': len(val_in_norm),
        'test_samples': len(test_in_norm),
        'normalization_method': pre.method,
        'input_mean': pre.input_scaler.mean_,
        'input_std': pre.input_scaler.scale_,
        'output_mean': pre.output_scaler.mean_,
        'output_std': pre.output_scaler.scale_
    }
    
    with open(processed_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nData saved to {processed_dir} directory")

if __name__ == "__main__":
    main()


