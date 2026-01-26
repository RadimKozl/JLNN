#!/usr/bin/env python3

"""
PyTorch Export Module for JLNN Models

This module provides utilities to convert JAX/Flax NNX models to PyTorch format
using ONNX as an intermediate representation. The conversion pipeline:
JAX/NNX → ONNX → PyTorch

Dependencies:
    - torch: PyTorch framework
    - onnx: ONNX model format
    - onnx2pytorch: ONNX to PyTorch converter (https://github.com/Talmaj/onnx2pytorch)

Installation:
    pip install torch onnx onnx2pytorch
"""

# Imports
import os
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Dict
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
    Converts a JLNN (JAX/NNX) model to a PyTorch module using ONNX as an intermediary.
    
    This function implements a three-stage conversion pipeline:
    1. Export the JAX/Flax model to ONNX format (using export_to_onnx)
    2. Load the ONNX model and convert it to PyTorch computational graph
    3. Optionally clean up the temporary ONNX file
    
    The resulting PyTorch module preserves:
    - Model architecture and computational graph
    - Trained weights and parameters
    - Łukasiewicz logic operations (as equivalent PyTorch ops)
    - Input/output shapes and dtypes
    
    Args:
        model (nnx.Module): The trained JLNN model to convert.
        sample_input (jnp.ndarray): Sample input tensor for tracing.
                                   Shape should match expected input format,
                                   typically [..., 2] for truth intervals.
        tmp_path (str): Temporary file path for intermediate ONNX file.
                       Default: "tmp_model.onnx"
        cleanup (bool): If True, removes the temporary ONNX file after conversion.
                       Set to False if you want to inspect the ONNX model.
                       Default: True
    
    Returns:
        torch.nn.Module: A PyTorch module with identical logic and weights.
                        Can be used for inference, fine-tuning, or integration
                        into PyTorch pipelines.
    
    Raises:
        ImportError: If required dependencies (torch, onnx, onnx2pytorch) 
                    are not installed.
        RuntimeError: If ONNX export or PyTorch conversion fails.
    
    References:
        - onnx2pytorch: https://github.com/Talmaj/onnx2pytorch
        - ONNX: https://onnx.ai/
        - PyTorch: https://pytorch.org/
    
    Example:
        >>> # Export JAX model to PyTorch
        >>> jax_model = MyJLNNModel(feature_dim=10)
        >>> sample = jnp.array([[0.3, 0.7], [0.1, 0.9]])
        >>> pytorch_model = export_to_pytorch(jax_model, sample)
        >>> 
        >>> # Use the PyTorch model
        >>> torch_input = torch.tensor([[0.3, 0.7], [0.1, 0.9]])
        >>> output = pytorch_model(torch_input)
        >>> 
        >>> # Save the PyTorch model
        >>> torch.save(pytorch_model.state_dict(), 'model.pth')
    
    Note:
        The conversion quality depends on:
        - Completeness of the ONNX export (some JAX ops may not map perfectly)
        - onnx2pytorch's support for ONNX operators
        - Complexity of logical operations in the model
        
        For production use, validate the converted model's outputs against
        the original JAX model on representative test data.
    """
    # Dependency check
    if not PYTORCH_AVAILABLE:
        raise ImportError(
            "PyTorch export dependencies are not installed.\n"
            "Please install required packages:\n"
            "  pip install torch onnx onnx2pytorch\n\n"
            "References:\n"
            "  - PyTorch: https://pytorch.org/get-started/locally/\n"
            "  - onnx2pytorch: https://github.com/Talmaj/onnx2pytorch"
        )
    
    print("=" * 80)
    print("JLNN MODEL EXPORT TO PYTORCH")
    print("=" * 80)
    
    # Stage 1: Export to ONNX
    print("\nStage 1/3: Exporting JAX model to ONNX...")
    try:
        export_to_onnx(model, sample_input, tmp_path)
        print(f"  ✓ ONNX export successful: {tmp_path}")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {str(e)}") from e
    
    # Stage 2: Load ONNX and convert to PyTorch
    print("\nStage 2/3: Converting ONNX to PyTorch...")
    try:
        onnx_model = onnx.load(tmp_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model loaded and validated")
        
        pytorch_model = ConvertModel(onnx_model)
        print(f"  ✓ PyTorch model created")
        
        # Set to evaluation mode by default
        pytorch_model.eval()
        
    except Exception as e:
        if cleanup and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"PyTorch conversion failed: {str(e)}") from e
    
    # Stage 3: Cleanup
    print("\nStage 3/3: Cleanup...")
    if cleanup:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"  ✓ Temporary ONNX file removed: {tmp_path}")
    else:
        print(f"  ⓘ ONNX file preserved at: {tmp_path}")
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"PyTorch model ready for inference or fine-tuning.")
    print(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    
    return pytorch_model


def verify_pytorch_conversion(
    jax_model: nnx.Module,
    pytorch_model: 'torch.nn.Module',
    sample_input: jnp.ndarray,
    tolerance: float = 1e-5
) -> Dict[str, any]:
    """
    Verifies that the PyTorch conversion produces equivalent outputs to the JAX model.
    
    This function runs both models on the same input and compares their outputs
    to ensure the conversion preserved the model's behavior.
    
    Args:
        jax_model (nnx.Module): Original JAX/Flax model.
        pytorch_model (torch.nn.Module): Converted PyTorch model.
        sample_input (jnp.ndarray): Test input tensor.
        tolerance (float): Maximum allowed difference between outputs.
    
    Returns:
        Dict containing:
            - 'passed': bool indicating if verification passed
            - 'max_diff': maximum absolute difference
            - 'mean_diff': mean absolute difference
            - 'jax_output': JAX model output
            - 'pytorch_output': PyTorch model output (as numpy)
    
    Example:
        >>> pytorch_model = export_to_pytorch(jax_model, sample)
        >>> results = verify_pytorch_conversion(jax_model, pytorch_model, sample)
        >>> if results['passed']:
        >>>     print("✓ Conversion verified successfully!")
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available for verification")
    
    print("\n" + "=" * 80)
    print("VERIFYING PYTORCH CONVERSION")
    print("=" * 80)
    
    # Run JAX model
    print("\nRunning JAX model...")
    jax_output = jax_model(sample_input)
    print(f"  JAX output shape: {jax_output.shape}")
    print(f"  JAX output dtype: {jax_output.dtype}")
    
    # Run PyTorch model
    print("\nRunning PyTorch model...")
    pytorch_model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(jnp.array(sample_input))
        pytorch_output = pytorch_model(torch_input)
        pytorch_output_np = pytorch_output.numpy()
    
    print(f"  PyTorch output shape: {pytorch_output_np.shape}")
    print(f"  PyTorch output dtype: {pytorch_output_np.dtype}")
    
    # Compare outputs
    print("\nComparing outputs...")
    diff = jnp.abs(jax_output - pytorch_output_np)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    
    print(f"  Maximum absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Tolerance threshold: {tolerance:.2e}")
    
    passed = max_diff < tolerance
    
    if passed:
        print("\n✓ VERIFICATION PASSED")
        print(f"  Outputs match within tolerance ({tolerance:.2e})")
    else:
        print("\n⚠ VERIFICATION FAILED")
        print(f"  Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
        warnings.warn(
            f"PyTorch conversion verification failed. "
            f"Max difference: {max_diff:.2e}, tolerance: {tolerance:.2e}"
        )
    
    print("=" * 80 + "\n")
    
    return {
        'passed': passed,
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'jax_output': jax_output,
        'pytorch_output': pytorch_output_np
    }


def generate_pytorch_state_dict_code(
    model: nnx.Module, 
    sample_input: jnp.ndarray,
    output_file: Optional[str] = None
) -> str:
    """
    Generates Python code for manually loading weights into a PyTorch model.
    
    This function is useful when you want to:
    - Port the model architecture manually to PyTorch
    - Inspect the exact weight values
    - Create custom PyTorch modules with pretrained JLNN weights
    
    Args:
        model (nnx.Module): The JLNN model to extract weights from.
        sample_input (jnp.ndarray): Sample input for model tracing.
        output_file (Optional[str]): If provided, saves the generated code to this file.
    
    Returns:
        str: Python code snippet that creates a PyTorch state_dict.
    
    Example:
        >>> code = generate_pytorch_state_dict_code(jax_model, sample)
        >>> print(code)
        >>> # Or save to file
        >>> generate_pytorch_state_dict_code(jax_model, sample, "weights.py")
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available for state dict generation")
    
    print("Generating PyTorch state_dict code...")
    
    # Convert model to PyTorch
    pytorch_model = export_to_pytorch(model, sample_input, cleanup=True)
    state_dict = pytorch_model.state_dict()
    
    # Generate code
    code_lines = [
        "#!/usr/bin/env python3",
        '"""',
        "Auto-generated PyTorch State Dictionary for JLNN Model",
        "",
        "This file contains the trained weights from a JAX/Flax JLNN model",
        "converted to PyTorch format. Load these weights into your PyTorch model:",
        "",
        "    model = YourPyTorchModel()",
        "    model.load_state_dict(state_dict)",
        '"""',
        "",
        "import torch",
        "",
        "# Model state dictionary",
        "state_dict = {",
    ]
    
    # Add each parameter
    for key, value in state_dict.items():
        value_np = value.cpu().numpy()
        shape_str = f"  # Shape: {tuple(value_np.shape)}, dtype: {value_np.dtype}"
        code_lines.append(shape_str)
        code_lines.append(f"    '{key}': torch.tensor({value_np.tolist()}),")
        code_lines.append("")
    
    code_lines.append("}")
    code_lines.append("")
    code_lines.append("# Usage example:")
    code_lines.append("# model = YourPyTorchModel()")
    code_lines.append("# model.load_state_dict(state_dict)")
    code_lines.append("# model.eval()")
    
    code = "\n".join(code_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"✓ State dict code saved to: {output_file}")
    
    print(f"✓ Generated code for {len(state_dict)} parameters")
    
    return code


def export_pytorch_with_verification(
    model: nnx.Module,
    sample_input: jnp.ndarray,
    output_path: str,
    tolerance: float = 1e-5
) -> 'torch.nn.Module':
    """
    Complete workflow: export to PyTorch and verify the conversion.
    
    This convenience function combines export and verification in a single call.
    
    Args:
        model (nnx.Module): JAX model to export.
        sample_input (jnp.ndarray): Sample input for tracing.
        output_path (str): Path to save the PyTorch model (.pth or .pt).
        tolerance (float): Verification tolerance for output comparison.
    
    Returns:
        torch.nn.Module: Verified PyTorch model.
    
    Raises:
        RuntimeError: If verification fails.
    
    Example:
        >>> pytorch_model = export_pytorch_with_verification(
        ...     jax_model, sample, "model.pth", tolerance=1e-5
        ... )
    """
    # Export
    pytorch_model = export_to_pytorch(model, sample_input)
    
    # Verify
    results = verify_pytorch_conversion(model, pytorch_model, sample_input, tolerance)
    
    if not results['passed']:
        raise RuntimeError(
            f"PyTorch conversion verification failed. "
            f"Max difference: {results['max_diff']:.2e}, "
            f"tolerance: {tolerance:.2e}"
        )
    
    # Save
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"\n✓ Verified PyTorch model saved to: {output_path}")
    
    return pytorch_model