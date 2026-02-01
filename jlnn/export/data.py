#!/usr/bin/env python3
"""
Logic parameter extraction utilities for model portability and symbolic reasoning.

This module provides functions to traverse the Flax NNX module hierarchy and 
extract learned logical parameters (weights, biases, thresholds) into standard 
Python dictionaries. This allows for saving model metadata as JSON or 
exporting the neural logic back into symbolic rules.
"""

# Imports
import jax.numpy as jnp
from flax import nnx
from typing import Dict, Any, Union
from jlnn.nn.gates import (
    WeightedAnd, WeightedOr, WeightedXor, 
    WeightedNand, WeightedNor, WeightedNot, WeightedImplication
)

def export_module_recursive(module: Any, name: str = "root") -> Dict[str, Any]:
    """
    Recursively extracts logical parameters while preserving the module hierarchy.

    This function traverses the Flax NNX module tree, identifying logical gates 
    and capturing their specific parameters. It maintains the tree structure, 
    allowing for the reconstruction of nested logical expressions (e.g., 
    reduction trees in n-ary gates).

    Args:
        module (Any): An instance of an NNX Module or a sub-gate to be inspected.
        name (str): The attribute name of the module within its parent. 
            Defaults to "root".

    Returns:
        Dict[str, Any]: A dictionary containing the gate's name, type, 
            extracted parameters (weights, beta, etc.), and optional nested sub-gates.
    """
    data = {
        "name": name,
        "type": module.__class__.__name__,
        "params": {}
    }

    # Use [...] for NNX Variable access to avoid DeprecationWarnings
    # weights (w >= 1.0) represent the relative importance of antecedents
    if hasattr(module, "weights"):
        data["params"]["weights"] = module.weights[...].tolist()
    
    # beta (b) is the threshold parameter for the Lukasiewicz t-norm
    if hasattr(module, "beta"):
        # Ensure beta is a standard float for JSON compatibility
        data["params"]["beta"] = float(module.beta[...])
        
    # Specific threshold for XOR gates (separating True from Contradictory/False)
    xor_param = "xor_threshold"
    if hasattr(module, xor_param):
        data["params"][xor_param] = module.xor_threshold[...].tolist()

    # Recursive Tree Traversal
    # We inspect the object's dictionary to find nested NNX modules.
    sub_modules = {}
    for attr_name, attr_value in vars(module).items():
        if isinstance(attr_value, nnx.Module):
            sub_modules[attr_name] = export_module_recursive(attr_value, attr_name)
    
    if sub_modules:
        data["sub_gates"] = sub_modules

    return data

def extract_logic_parameters(model: nnx.Module) -> Dict[str, Any]:
    """
    Primary API to extract the full logical structure and weights of a JLNN model.

    This function produces a portable, structured representation of the entire 
    logical network. The resulting dictionary is suitable for JSON serialization 
    and can be used by external symbolic reasoners to verify or visualize 
    learned rules.

    Args:
        model (nnx.Module): The top-level Flax NNX logical model instance.

    Returns:
        Dict[str, Any]: A nested dictionary containing framework metadata 
            and the recursive gate structure of the model.
    """
    return {
        "metadata": {
            "framework": "JLNN", 
            "version": "0.1.0"
        },
        "structure": export_module_recursive(model)
    }