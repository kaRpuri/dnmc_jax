import numpy as np

# Load the .npz file
filename = 'data_record_20deg.npz'  # Replace with your file path
data = np.load(filename)

# List all arrays stored in the file
print("Arrays in the .npz file:", data.files)

# Explore each array
for key in data.files:
    arr = data[key]
    print(f"\nArray '{key}':")
    print(f"  Shape: {arr.shape}")
    print(f"  Data type: {arr.dtype}")
    print(f"  First 5 elements:\n{arr.flat[:50]}")


