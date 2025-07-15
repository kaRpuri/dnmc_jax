import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple

def plot_inputs(data: Dict, show: bool = True, save_path: Optional[str] = None):
    """Plot control inputs (acceleration and steering rate)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    t = data["t_Ref"]
    
    # Plot acceleration
    ax1.plot(t, data["u_Input"].U_ax, color='black')
    ax1.set_ylabel('ax')
    
    # Plot steering rate
    ax2.plot(t, data["u_Input"].U_vdelta, color='black')
    ax2.set_ylabel('vdelta')
    ax2.set_xlabel('time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig

def plot_comparison(tsteps: np.ndarray, true_data: np.ndarray, 
                  predicted_data: np.ndarray, vline_pos: Optional[float] = None,
                  show: bool = True, save_path: Optional[str] = None):
    """Plot comparison of true vs predicted states"""
    state_names = ['x', 'y', 'ψ', 'δ', 'v', 'β', 'ψ̇']
    
    fig, axes = plt.subplots(7, 1, figsize=(10, 14))
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.plot(tsteps, true_data[i, :], 'k-', linewidth=1.5, label='data')
        ax.plot(tsteps, predicted_data[i, :], 'r-', linewidth=1.5, label='model')
        
        if vline_pos is not None:
            ax.axvline(x=vline_pos, color='black', linestyle='--')
            
        ax.set_ylabel(name)
        
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig