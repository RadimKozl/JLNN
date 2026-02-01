#!/usr/bin/env python3
"""
PyTorch Export Module for JLNN Models.

This module provides utilities to bridge the JAX/Flax NNX ecosystem with PyTorch.
It uses ONNX as an intermediate representation to translate logical neural 
network structures into PyTorch-compatible modules.
"""

# Imports
import os
import numpy as np
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Dict, Any
import warnings

# Import our ONNX export function
from jlnn.export.onnx import export_to_onnx

# Optional PyTorch imports
try:
    import torch
    import onnx
    from onnx2pytorch import ConvertModel
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    onnx = None
    ConvertModel = None
    PYTORCH_AVAILABLE = False


def export_to_pytorch(
    model: nnx.Module, 
    sample_input: jnp.ndarray, 
    tmp_path: str = "tmp_model.onnx",
    cleanup: bool = True
) -> 'torch.nn.Module':
    """
    Converts a JLNN model to a PyTorch Module via ONNX.

    The conversion process involves tracing the JAX model to an ONNX graph
    and then re-mapping those operations to PyTorch layers using onnx2pytorch.

    Args:
        model (nnx.Module): The logical neural network (NNX) to export.
        sample_input (jnp.ndarray): Representative input for shape tracing (batch, 2).
        tmp_path (str): Temporary path for the intermediate ONNX file.
        cleanup (bool): If True, deletes the temporary ONNX file after conversion.

    Returns:
        torch.nn.Module: An equivalent PyTorch module for inference.

    Raises:
        ImportError: If torch, onnx, or onnx2pytorch are not installed.
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError(
            "PyTorch or onnx2pytorch not found. Install with: "
            "pip install torch onnx onnx2pytorch"
        )

    print("\n" + "=" * 80)
    print("JLNN MODEL EXPORT TO PYTORCH")
    print("=" * 80)

    try:
        # Stage 1: JAX -> ONNX
        print("\nStage 1/3: Exporting JAX model to ONNX...")
        export_to_onnx(model, sample_input, tmp_path)
        
        # Stage 2: ONNX -> PyTorch
        print("\nStage 2/3: Converting ONNX to PyTorch...")
        onnx_model = onnx.load(tmp_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model loaded and validated")
        
        pytorch_model = ConvertModel(onnx_model)
        print("  ✓ PyTorch model created")
        
        return pytorch_model

    finally:
        # Stage 3: Cleanup
        if cleanup and os.path.exists(tmp_path):
            print("\nStage 3/3: Cleanup...")
            os.remove(tmp_path)
            print(f"  ✓ Temporary ONNX file removed: {tmp_path}")
            print("\n" + "=" * 80)
            print("CONVERSION COMPLETE")
            print("=" * 80)


def verify_pytorch_conversion(
    jax_model: nnx.Module,
    pytorch_model: 'torch.nn.Module',
    sample_input: jnp.ndarray,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Verifies numerical consistency between the original JAX model and the exported PyTorch model.

    This utility performs a forward pass on both models using the same input and 
    compares the resulting truth intervals using the specified absolute tolerance.

    Args:
        jax_model (nnx.Module): The original source model.
        pytorch_model (torch.nn.Module): The converted destination model.
        sample_input (jnp.ndarray): Input data for comparison.
        tolerance (float): Maximum allowed absolute difference between outputs.

    Returns:
        Dict[str, Any]: A report containing:
            - 'passed' (bool): Whether the difference is within tolerance.
            - 'max_diff' (float): The maximum observed absolute error.
            - 'jax_output' (np.ndarray): Output from the original model.
            - 'pytorch_output' (np.ndarray): Output from the converted model.
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available for verification")

    # Run JAX model
    jax_output = jax_model(sample_input)
    
    # Run PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        # JAX -> NumPy -> Torch bridge
        numpy_input = np.array(sample_input).astype(np.float32)
        torch_input = torch.from_numpy(numpy_input)
        pytorch_output_raw = pytorch_model(torch_input)
        pytorch_output = pytorch_output_raw.cpu().numpy()

    # Compare
    jax_out_np = np.array(jax_output)
    diff = np.abs(jax_out_np - pytorch_output)
    max_diff = float(np.max(diff))
    passed = max_diff <= tolerance

    return {
        'passed': passed,
        'max_diff': max_diff,
        'jax_output': jax_out_np,
        'pytorch_output': pytorch_output
    }