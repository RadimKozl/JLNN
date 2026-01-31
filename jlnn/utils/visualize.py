#!/usr/bin/env python3

"""
Visualization tools for inspecting LNN state and learned logic.
"""

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from jlnn.core import intervals
from typing import Dict, List, Optional

def plot_truth_intervals(
    intervals_dict: Dict[str, jnp.ndarray], 
    title: str = "JLNN Truth Intervals",
    show: bool = True
) -> plt.Figure:
    """
    Renders a horizontal bar chart of truth intervals for model state inspection.

    In Logical Neural Networks, truth is represented by an interval [L, U]. 
    This visualization maps these intervals to horizontal bars:
    - The left edge represents the Lower bound (necessary truth).
    - The right edge represents the Upper bound (possible truth).
    - The width of the bar indicates uncertainty (ignorance).
    - A collapsed bar (L ≈ U) represents a precise classical truth value.

    The function automatically performs a consistency check: if L > U, the bar 
    is rendered in red to indicate a 'Logical Contradiction', signifying that 
    the network has reached an unsatisfiable state where evidence for truth 
    exceeds evidence for possibility.

    Args:
        intervals_dict: Dictionary mapping symbolic names (predicates/gates) 
            to JAX arrays of shape (2,) or (batch, 2). If batched, 
            the first sample is typically visualized.
        title: Title of the plot, identifying the model or inference step.
        show: If True, calls plt.show(). Disable this for automated testing 
              or when further figure manipulation is required.

    Returns:
        The matplotlib Figure object for further customization or logging.
    """
    names = list(intervals_dict.keys())
    values = list(intervals_dict.values())
    
    lowers = [float(intervals.get_lower(v)) for v in values]
    uppers = [float(intervals.get_upper(v)) for v in values]
    
    fig, ax = plt.subplots(figsize=(10, max(2, len(names) * 0.4)))
    
    for i, (name, l, u) in enumerate(zip(names, lowers, uppers)):
        is_contradictory = l > u
        color = 'red' if is_contradictory else 'skyblue'
        
        if not is_contradictory:
            ax.barh(i, u - l, left=l, color=color, edgecolor='black', alpha=0.7)
        else:
            ax.barh(i, l - u, left=u, color=color, edgecolor='black', alpha=0.7)
            
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Truth Value Interval [Lower, Upper]")
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if show:
        plt.show()
    return fig
    

def plot_gate_weights(
    weights: jnp.ndarray, 
    input_labels: List[str], 
    gate_name: str = "Gate",
    show: bool = True
) -> plt.Figure:
    """
    Generates a heatmap to visualize the learned importance of inputs for a specific gate.

    In Logical Neural Networks, weights (constrained to w >= 1.0) act as attention 
    mechanisms over logical antecedents. A higher weight indicates that the 
    corresponding input has a stronger influence on the gate's activation 
    and the overall truth value of the formula.

    Args:
        weights: A JAX array of trained weights from the gate module.
        input_labels: Symbolic names of the input predicates (e.g., from metadata).
        gate_name: The label of the logical gate being inspected (e.g., 'WeightedAND_1').
        show: If True, displays the plot immediately. Set to False for 
              programmatic use or automated testing.

    Returns:
        The matplotlib Figure object containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    data = weights.reshape(1, -1)
    sns.heatmap(data, annot=True, xticklabels=input_labels, yticklabels=[gate_name], 
                cmap="YlGnBu", ax=ax)
    plt.title(f"Learned Weight Importance for {gate_name}")
    plt.xlabel("Input Predicates")
    
    if show:
        plt.show()
    return fig

def plot_training_log_loss(
    losses: List[float], 
    title: str = "Training Convergence",
    show: bool = True
) -> plt.Figure:
    """
    Plots the loss curve to visualize the optimization and logical grounding process.
    
    In Logical Neural Networks, the loss trajectory reflects how well the model 
    is satisfying logical constraints while fitting the data. Monitoring this 
    convergence is crucial for identifying 'over-constrained' models or 
    oscillations caused by conflicting logical rules.

    Args:
        losses: A list or array of loss values recorded during training epochs.
        title: Descriptive title for the plot (e.g., 'Convergence: XOR Problem').
        show: If True, invokes the backend's display (GUI or Notebook inline). 
              Set to False for automated reporting or background processing.

    Returns:
        The matplotlib Figure object representing the convergence visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, label='Loss', color='tab:red', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epoch / Iteration')
    ax.set_ylabel('Loss Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()
    return fig