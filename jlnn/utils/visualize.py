#!/usr/bin/env python3

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jlnn.core import intervals
from typing import Dict, List, Optional

def plot_truth_intervals(intervals_dict: Dict[str, jnp.ndarray], title: str = "JLNN Truth Intervals"):
    """
    Renders a horizontal bar chart of truth intervals for model inspection.
    
    Each bar represents the range [L, U]. If a contradiction is detected (L > U), 
    the bar is highlighted in red to signal a logical conflict in the network.
    
    Args:
        intervals_dict (Dict[str, jnp.ndarray]): Mapping of gate/predicate names 
                                                 to their [L, U] interval arrays.
        title (str): The chart title.
    """
    names = list(intervals_dict.keys())
    values = list(intervals_dict.values())
    
    lowers = [float(intervals.get_lower(v)) for v in values]
    uppers = [float(intervals.get_upper(v)) for v in values]
    
    fig, ax = plt.subplots(figsize=(10, max(2, len(names) * 0.4)))
    
    for i, (name, l, u) in enumerate(zip(names, lowers, uppers)):
        is_contradictory = l > u
        color = 'red' if is_contradictory else 'skyblue'
        
        # Draw the interval line
        ax.hlines(i, min(l, u), max(l, u), colors=color, lw=6, alpha=0.7)
        # Draw bound markers
        ax.plot(l, i, 'o', color='navy', label='Lower (L)' if i == 0 else "")
        ax.plot(u, i, 's', color='darkorange', label='Upper (U)' if i == 0 else "")
        
        if is_contradictory:
            ax.text(1.05, i, f"CONFLICT! ({l-u:.2f})", color='red', va='center')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel("Truth Value [0, 1]")
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_gate_weights(weights: jnp.ndarray, input_labels: List[str], gate_name: str = "Gate"):
    """
    Generates a heatmap to visualize the importance of each input for a specific gate.

    In JLNN, weights (w >= 1.0) signify the relative importance of an antecedent. 
    A higher weight means the model relies more heavily on that specific truth 
    value to reach a logical conclusion.

    Args:
        weights (jnp.ndarray): Array of trained weights from a gate module.
        input_labels (List[str]): Symbolic names for the input predicates (from metadata).
        gate_name (str): Name of the gate being visualized.
    """
    plt.figure(figsize=(8, 6))
    data = weights.reshape(1, -1)
    sns.heatmap(data, annot=True, xticklabels=input_labels, yticklabels=[gate_name], cmap="YlGnBu")
    plt.title(f"Learned Weight Importance for {gate_name}")
    plt.xlabel("Input Predicates")
    plt.show()

def plot_training_log_loss(losses: List[float], title: str = "Training Convergence"):
    """
    Plots the loss curve over training iterations.
    
    Useful for monitoring how the logical constraints and truth values 
    stabilize over time.
    """
    plt.plot(losses, label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()