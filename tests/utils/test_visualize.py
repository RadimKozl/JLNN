#!/usr/bin/env python3
"""
Smoke tests for visualization utilities in the JLNN framework.

These tests ensure that all plotting functions (truth intervals, gate weights, 
and training convergence) execute without runtime errors when provided with 
valid data. To prevent blocking and warnings in CI/CD environments, 
a non-interactive Matplotlib backend is used.
"""

import pytest
import jax.numpy as jnp
import matplotlib
# Use Agg backend to avoid 'no display' errors and warnings in headless environments
matplotlib.use('Agg')
from jlnn.utils.visualize import plot_truth_intervals, plot_gate_weights, plot_training_log_loss

def test_visualize_smoke():
    """
    Verifies that all visualization tools are operational and handle data correctly.
    
    By setting 'show=False', the test validates the figure generation logic 
    (Matplotlib/Seaborn) while suppressing 'non-interactive backend' warnings.
    """
    try:
        # 1. Test Truth Intervals Visualization
        # Covers both normal intervals and contradictions (color logic)
        interval_data = {
            "Predicate_A": jnp.array([0.2, 0.8]), 
            "Contradiction_B": jnp.array([0.9, 0.1])
        }
        plot_truth_intervals(interval_data, title="Smoke Test Intervals", show=False)
        
        # 2. Test Gate Weights Heatmap
        # Verifies the heatmap rendering for learned logic importance
        weights = jnp.array([1.5, 2.0, 1.1])
        labels = ["input_x", "input_y", "input_z"]
        plot_gate_weights(weights, labels, gate_name="WeightedAND_Test", show=False)
        
        # 3. Test Training Loss Curve
        # Verifies the convergence plot utility
        sample_losses = [1.0, 0.7, 0.5, 0.35, 0.3]
        plot_training_log_loss(sample_losses, title="Loss Smoke Test", show=False)
        
    except Exception as e:
        pytest.fail(f"Visualization utility crashed during smoke test: {e}")

if __name__ == "__main__":
    # Allows manual execution of the smoke test
    test_visualize_smoke()
    print("All visualization smoke tests passed successfully.")