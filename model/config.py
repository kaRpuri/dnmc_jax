from pathlib import Path

CONFIG = {
    'batch_size': 256,
    'epochs': 100,
    'learning_rate': 1e-3,
    'beta1': 0.5,
    'beta2': 0.1,
    'beta3': 0.0001,
    'save_dir': './checkpoints',
    'project': 'vehicle-dynamics',
    'entity': None,  # Set your wandb entity if needed
    'run_name': None,  # Optionally set a custom run name
    'seed': 42,
    'train_data_path': '../data_processing/processed/train.npz',
    'val_data_path': '../data_processing/processed/val.npz'
}