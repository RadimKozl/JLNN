#!/usr/bin/env python3
"""
Integration utilities for Xarray data structures.

This module provides tools to bridge JAX tensors with xarray, allowing 
symbolic labeling of neural outputs and trained logical weights.
"""

# Imports
import xarray as xr
import jax.numpy as jnp
from typing import Any, Dict, List

def model_to_xarray(gate_outputs: Dict[str, jnp.ndarray], sample_labels: List[str]) -> xr.Dataset:
    """
    Converts logical model outputs into a labeled xarray Dataset.

    By mapping raw (batch, 2) tensors to a Dataset with 'sample' and 'bound' 
    dimensions, we enable powerful scientific indexing and visualization 
    of truth intervals.

    Args:
        gate_outputs: Mapping of gate names to their [L, U] output tensors.
        sample_labels: Names for the samples in the batch (e.g., individual IDs).

    Returns:
        xr.Dataset: Multi-dimensional dataset containing truth intervals.
    """
    ds_dict = {}
    for name, data in gate_outputs.items():
        ds_dict[name] = xr.DataArray(
            data,
            dims=["sample", "bound"],
            coords={
                "sample": sample_labels,
                "bound": ["Lower", "Upper"]
            },
            name=name
        )
    return xr.Dataset(ds_dict)

def extract_weights_to_xarray(weights: jnp.ndarray, input_labels: List[str], gate_name: str) -> xr.DataArray:
    """
    Extracts trained logic weights into a labeled xarray DataArray.

    In LNN, weights w >= 1.0 represent the importance of an antecedent. 
    This function creates a structured view of these weights, mapping 
    each value to its symbolic predicate name.

    Args:
        weights: JAX array of weights from a logical gate.
        input_labels: Symbolic names of the inputs (e.g., ['A', 'B']).
        gate_name: Name of the gate these weights belong to.

    Returns:
        xr.DataArray: Labeled weights with 'input' dimension.
    """
    return xr.DataArray(
        weights,
        dims=["input"],
        coords={"input": input_labels},
        name=f"weights_{gate_name}"
    )