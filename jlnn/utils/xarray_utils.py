#!/usr/bin/env python3

# Imports
import xarray as xr
import jax.numpy as jnp
from typing import Any, Dict, List

def model_to_xarray(gate_outputs: Dict[str, jnp.ndarray], predicate_names: List[str]) -> xr.Dataset:
    """
    Converts logical model outputs into a labeled xarray Dataset for advanced analysis.

    This utility bridges the gap between raw JAX tensors and structured scientific data. 
    By labeling dimensions such as 'predicate', 'bound' (Lower/Upper), and 'sample', 
    researchers can use xarray's powerful indexing to inspect the model's reasoning.

    Args:
        gate_outputs (Dict[str, jnp.ndarray]): Mapping of gate names to their output tensors 
            of shape (batch, 2).
        predicate_names (List[str]): List of labels for the dimension coordinates.

    Returns:
        xr.Dataset: A multi-dimensional dataset containing truth intervals.
    """
    ds_dict = {}
    
    for name, data in gate_outputs.items():
        # data is expected to be (batch, 2)
        ds_dict[name] = xr.DataArray(
            data,
            dims=["sample", "bound"],
            coords={"bound": ["Lower", "Upper"]},
            name=name
        )
        
    return xr.Dataset(ds_dict)

def extract_weights_to_xarray(model_params: Dict[str, Any]) -> xr.DataArray:
    """
    Extracts flattened weights and maps them to a labeled DataArray.
    
    This is useful for cross-referencing learned weights across different 
    training runs or experimental configurations.
    """
    # Logic to filter and label weights from the structured dictionary
    # Useful for analyzing WeightedXor reduction trees.
    pass