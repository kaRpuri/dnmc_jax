import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "processed"
NPZ_FILES = ["train.npz", "val.npz", "test.npz"]

def load_npz(file_path):
    data = np.load(file_path)
    return dict(data)

def describe_array(arr, name):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, mean={arr.mean():.4f}, std={arr.std():.4f}")

def plot_arrays(data, file_label):
    plt.figure(figsize=(15, 8))
    keys = list(data.keys())
    for i, key in enumerate(keys):
        arr = data[key]
        if arr.ndim == 1:
            plt.subplot(len(keys), 1, i+1)
            plt.plot(arr)
            plt.title(f"{file_label} - {key}")
            plt.tight_layout()
        elif arr.ndim == 2 and arr.shape[1] < 10:
            for j in range(arr.shape[1]):
                plt.subplot(len(keys), 1, i+1)
                plt.plot(arr[:, j], label=f"{key}[{j}]")
            plt.title(f"{file_label} - {key}")
            plt.legend()
            plt.tight_layout()
    plt.show()

def plot_inputs(inputs, file_label):
    plt.figure(figsize=(15, 6))
    if inputs.ndim == 2:
        for i in range(inputs.shape[1]):
            plt.plot(inputs[:, i], label=f"input[{i}]")
        plt.title(f"{file_label} - Inputs")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif inputs.ndim == 1:
        plt.plot(inputs)
        plt.title(f"{file_label} - Inputs")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

def main():
    for fname in NPZ_FILES:
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            print(f"File not found: {fpath}")
            continue
        print(f"\n=== {fname} ===")
        data = load_npz(fpath)
        for k, v in data.items():
            describe_array(v, k)
        plot_arrays(data, fname)

        # Plot only the inputs if present
        if 'inputs' in data:
            plot_inputs(data['inputs'], fname)

        # Assume 'inputs' and 'targets' are present
        if 'inputs' in data and 'targets' in data:
            print(f"Inputs shape: {data['inputs'].shape}")
            print(f"Targets shape: {data['targets'].shape}")
            print(f"Inputs mean: {data['inputs'].mean(axis=0)}")
            print(f"Inputs std: {data['inputs'].std(axis=0)}")
            print(f"Targets mean: {data['targets'].mean(axis=0)}")
            print(f"Targets std: {data['targets'].std(axis=0)}")

if __name__ == "__main__":
    main()