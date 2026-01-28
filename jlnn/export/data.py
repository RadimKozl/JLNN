#!/usr/bin/env python3

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
    and capturing their specific parameters (weights, beta, offsets). By maintaining 
    the tree structure, it allows for the reconstruction of nested logical 
    expressions, such as those found in binary-reduction trees of n-ary XOR gates.

    Args:
        module (Any): An instance of an NNX Module or a sub-gate.
        name (str): The attribute name of the module within its parent. Defaults to "root".

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "name": String identifier of the gate.
            - "type": Class name of the gate (e.g., 'WeightedAnd').
            - "params": Dictionary of numerical values (weights, beta, etc.).
            - "sub_gates": (Optional) Nested dictionaries of child modules.
    """
    data = {
        "name": name,
        "type": module.__class__.__name__,
        "params": {}
    }

    # 1. Parameter Extraction Logic
    # We extract known LNN parameters and convert them from JAX arrays 
    # to standard Python types for JSON serializability.
    
    # Standard gate weights (AND, OR, NAND, NOR)
    if hasattr(module, 'weights'):
        data["params"]["weights"] = module.weights.value.tolist()
    
    # Sensitivity threshold (Bias equivalent)
    if hasattr(module, 'beta'):
        data["params"]["beta"] = float(module.beta.value)
    
    # Unary weight for negation gates
    if hasattr(module, 'weight'):
        data["params"]["weight"] = float(module.weight.value)
        
    # Specialized weights for complex gate compositions (e.g., WeightedXor sub-components)
    for xor_param in ['weights_or', 'weights_nand', 'weights_and']:
        if hasattr(module, xor_param):
            data["params"][xor_param] = getattr(module, xor_param).value.tolist()

    # 2. Recursive Tree Traversal
    # We inspect the object's dictionary to find nested NNX modules.
    # This captures left_child/right_child structures in recursive gates.
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
    logical network. The resulting dictionary can be saved as a JSON metadata 
    file or used by symbolic reasoners to translate the neural weights 
    back into formal logic rules.

    Args:
        model (nnx.Module): The top-level Flax NNX logical model.

    Returns:
        Dict[str, Any]: A nested dictionary containing framework metadata 
            and the recursive gate structure.
    """
    return {
        "metadata": {
            "framework": "JLNN", 
            "version": "1.1",
            "description": "Structured Logical Parameter Export"
        },
        "structure": export_module_recursive(model)
    }