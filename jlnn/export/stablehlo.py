#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jax.export import export
from typing import Any


def export_to_stablehlo(model: nnx.Module, sample_input: jnp.ndarray) -> jax.export.Exported:
    """
    Compiles the JLNN model into a StableHLO artifact for high-performance deployment.
    
    StableHLO is an operation set for deep learning compilers and is part of the 
    OpenXLA ecosystem (https://openxla.org/stablehlo). This export ensures that 
    the specific Łukasiewicz logic kernels defined in 'functional.py' are lowered 
    into highly optimized, hardware-agnostic HLO operations.
    
    The export process involves:
    1. Splitting the stateful NNX model into GraphDef (structure) and State (parameters)
    2. Creating a pure functional wrapper compatible with JAX tracing
    3. Generating abstract value specifications (avals) for all inputs
    4. Lowering to StableHLO via the JAX export API
    
    Args:
        model (nnx.Module): The logic-based neural network (Flax NNX) containing
            logical gates such as WeightedAnd, WeightedOr, etc.
        sample_input (jnp.ndarray): Input data representing truth intervals 
            (shape: [..., 2]) used to trace shapes and dtypes during compilation.
    
    Returns:
        jax.export.Exported: An object containing the StableHLO MLIR module 
            and serialized model state. Can be inspected via .mlir_module() or executed via .call().
    
    References:
        - StableHLO Specification: https://openxla.org/stablehlo
        - JAX Export Guide: https://docs.jax.dev/en/latest/jax.export.html
        - XLA Flags & Optimization: https://docs.jax.dev/en/latest/xla_flags.html
    
    Example:
        >>> model = MyJLNNModel(feature_dim=10)
        >>> sample = jnp.array([[0.3, 0.7], [0.1, 0.9]])
        >>> exported = export_to_stablehlo(model, sample)
        >>> # Inspect the StableHLO intermediate representation
        >>> print(exported.mlir_module())
        >>> # Execute the exported model
        >>> graphdef, state = nnx.split(model)
        >>> result = exported.call(state, sample)
    """
    # Step 1: State-Graph Separation
    # StableHLO requires pure functions without Python side-effects.
    # We decompose the NNX module into its static structure (graphdef) 
    # and dynamic parameters (state).
    graphdef, state = nnx.split(model)
    
    # Step 2: Pure Function Definition
    # JAX's export mechanism requires a JIT-compiled pure function.
    # This wrapper reconstructs the stateful model inside the XLA cluster.
    @jax.jit
    def forward_fn(state, x):
        # Reconstruct the stateful object within the JIT boundary
        m = nnx.merge(graphdef, state)
        return m(x)
    
    # Step 3: Abstract Value Preparation (Avals)
    # JAX export needs precise descriptions of input structures (shapes & dtypes).
    # For LNN truth intervals, we ensure the tracing captures the [..., 2] shape.
    state_avals = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state
    )
    input_avals = jax.ShapeDtypeStruct(sample_input.shape, sample_input.dtype)
    
    # Step 4: Lowering to StableHLO
    # This creates an MLIR (Multi-Level Intermediate Representation) module
    # containing StableHLO operations optimized for XLA compilation.
    # Reference: https://docs.jax.dev/en/latest/jax.export.html
    exported = export(forward_fn)(state_avals, input_avals)
    
    print("✓ StableHLO export successful. Model lowered to MLIR/StableHLO.")
    print(f"  Input shape: {sample_input.shape}, dtype: {sample_input.dtype}")
    
    return exported


def save_stablehlo_artifact(exported: jax.export.Exported, path: str):
    """
    Serializes the StableHLO module to a binary file for deployment.
    
    The serialized artifact is a portable representation of the compiled model
    that can be:
    - Loaded by XLA runtimes without Python/JAX dependencies
    - Converted to other formats (TFLite, ONNX via intermediate tools)
    - Deployed to cloud TPUs, GPUs, or custom accelerators
    - Used with OpenXLA toolchain for further optimization
    
    The serialization format preserves:
    - Complete StableHLO computation graph
    - Model parameters and their shapes
    - Type information and constant values
    - Control flow and conditional operations
    
    Args:
        exported (jax.export.Exported): The exported StableHLO model artifact
            returned by export_to_stablehlo().
        path (str): Destination file path for the serialized artifact.
            Convention: use .stablehlo or .mlir extension.
    
    References:
        - OpenXLA StableHLO: https://openxla.org/stablehlo
        - XLA Compilation: https://docs.jax.dev/en/latest/xla_flags.html
    
    Example:
        >>> exported = export_to_stablehlo(model, sample_input)
        >>> save_stablehlo_artifact(exported, "jlnn_model.stablehlo")
        >>> # Later, load and execute:
        >>> with open("jlnn_model.stablehlo", "rb") as f:
        >>>     serialized = f.read()
        >>> loaded = jax.export.deserialize(serialized)
    """
    # Serialize to MLIR bytecode format
    serialized_bytes = exported.serialize()
    
    with open(path, "wb") as f:
        f.write(serialized_bytes)
    
    size_kb = len(serialized_bytes) / 1024
    print(f"✓ StableHLO artifact saved to: {path}")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Format: MLIR bytecode (StableHLO dialect)")


def inspect_stablehlo_module(exported: jax.export.Exported, verbose: bool = False):
    """
    Inspects and prints the StableHLO MLIR representation for debugging.
    
    This function is useful for:
    - Verifying correct lowering of Łukasiewicz logic operations
    - Identifying optimization opportunities in the HLO graph
    - Debugging shape mismatches or type errors
    - Understanding the compiled computation structure
    
    Args:
        exported (jax.export.Exported): The exported StableHLO model.
        verbose (bool): If True, prints the full MLIR module.
            If False, prints only a summary.
    
    Example:
        >>> exported = export_to_stablehlo(model, sample_input)
        >>> inspect_stablehlo_module(exported, verbose=True)
    """
    mlir_str = exported.mlir_module()
    
    if verbose:
        print("=" * 80)
        print("FULL STABLEHLO MLIR MODULE")
        print("=" * 80)
        print(mlir_str)
        print("=" * 80)
    else:
        lines = mlir_str.split('\n')
        print("=" * 80)
        print("STABLEHLO MODULE SUMMARY")
        print("=" * 80)
        print(f"Total lines: {len(lines)}")
        print(f"First 20 lines:")
        print('\n'.join(lines[:20]))
        print("...")
        print(f"Last 10 lines:")
        print('\n'.join(lines[-10:]))
        print("=" * 80)
        print("Use verbose=True to see the full module.")


def export_workflow(
    model: nnx.Module, 
    sample_input: jnp.ndarray, 
    output_path: str,
    inspect: bool = False
) -> jax.export.Exported:
    """
    Complete workflow for exporting and saving a JLNN model to StableHLO.
    
    This convenience function combines export, inspection, and serialization
    in a single call for common deployment scenarios.
    
    Args:
        model (nnx.Module): The JLNN model to export.
        sample_input (jnp.ndarray): Sample input for tracing.
        output_path (str): Path where the StableHLO artifact will be saved.
        inspect (bool): If True, prints the MLIR module for debugging.
    
    Returns:
        jax.export.Exported: The exported model artifact.
    
    References:
        - StableHLO: https://openxla.org/stablehlo
        - JAX Export: https://docs.jax.dev/en/latest/jax.export.html
        - XLA Flags: https://docs.jax.dev/en/latest/xla_flags.html
    
    Example:
        >>> model = MyJLNNModel(feature_dim=10)
        >>> sample = jnp.array([[0.3, 0.7], [0.1, 0.9]])
        >>> exported = export_workflow(
        ...     model, sample, "model.stablehlo", inspect=True
        ... )
    """
    print("\n" + "=" * 80)
    print("JLNN MODEL EXPORT TO STABLEHLO")
    print("=" * 80 + "\n")
    
    # Step 1: Export to StableHLO
    print("Step 1: Exporting to StableHLO...")
    exported = export_to_stablehlo(model, sample_input)
    
    # Step 2: Optional inspection
    if inspect:
        print("\nStep 2: Inspecting MLIR module...")
        inspect_stablehlo_module(exported, verbose=False)
    
    # Step 3: Save artifact
    print("\nStep 3: Saving serialized artifact...")
    save_stablehlo_artifact(exported, output_path)
    
    # Step 4: Verification
    print("\nStep 4: Verifying exported model...")
    graphdef, state = nnx.split(model)
    
    # Test original model
    original_output = model(sample_input)
    
    # Test exported model
    exported_output = exported.call(state, sample_input)
    
    # Compare outputs
    max_diff = jnp.max(jnp.abs(original_output - exported_output))
    print(f"  Max difference between original and exported: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("  ✓ Verification passed: outputs match within tolerance")
    else:
        print(f"  ⚠ Warning: outputs differ by {max_diff:.2e}")
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80 + "\n")
    
    return exported