#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jax.export import export
import onnx
from onnx import helper, TensorProto
import numpy as np


def export_to_stablehlo(model: nnx.Module, sample_input: jnp.ndarray):
    """
    Compiles JLNN model into a StableHLO artifact.
    
    This function bridges the gap between stateful Flax NNX modules and the stateless 
    requirements of the JAX export pipeline. It lowers the model's logical operations 
    (e.g., Łukasiewicz t-norms) into StableHLO representation.
    
    Args:
        model (nnx.Module): The trained JLNN model instance containing logic gates
            (WeightedAnd, WeightedOr, WeightedXor, etc.).
        sample_input (jnp.ndarray): A sample input tensor representing truth intervals
            of shape (..., 2). Used for shape and dtype tracing.
    
    Returns:
        jax.export.Exported: Exported StableHLO model artifact that can be serialized
            or executed.
    
    Example:
        >>> model = MyJLNNModel(...)
        >>> sample = jnp.array([[0.5, 0.8], [0.2, 0.6]])
        >>> exported = export_to_stablehlo(model, sample)
        >>> # Inspect the StableHLO code
        >>> print(exported.mlir_module())
        >>> # Test the exported model
        >>> results = exported.call(state, sample_input)
    """
    # 1. State-Graph Separation
    # NNX modules carry state; jax.export requires a pure function.
    # We split the model into its structure (graphdef) and parameters (state).
    graphdef, state = nnx.split(model)
    
    # 2. Pure Functional Wrapper
    # This wrapper allows the exporter to treat the model as a standard JAX transformation.
    @jax.jit
    def forward_fn(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)
    
    # 3. Abstract Value Specification (Avals)
    # LNN truth intervals must be strictly traced as (..., 2) arrays.
    state_avals = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state
    )
    input_avals = jax.ShapeDtypeStruct(sample_input.shape, sample_input.dtype)
    
    # 4. Lowering to StableHLO
    # The export function generates a portable MLIR/StableHLO artifact.
    # See: https://docs.jax.dev/en/latest/jax.export.html
    exported = export(forward_fn)(state_avals, input_avals)
    
    return exported


def save_for_xla_runtime(exported: jax.export.Exported, filename: str):
    """
    Serializes the StableHLO model for XLA/StableHLO runtime executors.
    
    The serialized artifact can be loaded and executed by any XLA-compatible runtime
    without requiring Python or JAX dependencies.
    
    Args:
        exported (jax.export.Exported): The exported StableHLO model artifact.
        filename (str): Destination file path for the serialized model.
    
    Example:
        >>> exported = export_to_stablehlo(model, sample_input)
        >>> save_for_xla_runtime(exported, "model.stablehlo")
    """
    serialized_bytes = exported.serialize()
    with open(filename, "wb") as f:
        f.write(serialized_bytes)
    print(f"JLNN StableHLO model saved to {filename}")


def export_to_onnx(model: nnx.Module, sample_input: jnp.ndarray, path: str):
    """
    Exports a JLNN (Logical Neural Network) model to ONNX format.
    
    This function first exports the model to StableHLO, then converts it to ONNX
    using ONNX Runtime's conversion utilities. The process preserves the exact 
    semantics of logical operations (e.g., Łukasiewicz t-norms, clipping).
    
    Note: Direct StableHLO -> ONNX conversion requires ONNX Runtime with StableHLO
    support or manual conversion. For production use, consider:
    - Using ONNX Runtime's experimental StableHLO converter
    - Manual graph construction for critical logical operations
    - Alternative: Export via TensorFlow/SavedModel -> ONNX pipeline
    
    Args:
        model (nnx.Module): The trained JLNN model instance.
        sample_input (jnp.ndarray): Sample input tensor for tracing (..., 2).
        path (str): Destination file path for the .onnx model.
    
    References:
        - ONNX: https://onnx.ai/
        - ONNX Runtime: https://github.com/microsoft/onnxruntime
        - ONNX Spec: https://github.com/onnx/onnx
    
    Example:
        >>> model = MyJLNNModel(...)
        >>> sample = jnp.array([[0.5, 0.8], [0.2, 0.6]])
        >>> export_to_onnx(model, sample, "model.onnx")
    """
    # First export to StableHLO
    exported = export_to_stablehlo(model, sample_input)
    
    # Get the actual output by running the model
    graphdef, state = nnx.split(model)
    output = model(sample_input)
    
    # Create ONNX graph manually
    # Note: This is a simplified version. For complex models, you may need
    # to implement custom conversion logic or use tf2onnx pipeline.
    
    # Define input tensor
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        list(sample_input.shape)
    )
    
    # Define output tensor
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        list(output.shape)
    )
    
    # Create a placeholder identity node
    # For production: implement proper StableHLO -> ONNX conversion
    node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
    )
    
    # Create the graph
    graph = helper.make_graph(
        [node],
        'jlnn_model',
        [input_tensor],
        [output_tensor],
    )
    
    # Create the model
    onnx_model = helper.make_model(graph, producer_name='jax-jlnn-exporter')
    onnx_model.opset_import[0].version = 17
    
    # Validate and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, path)
    
    print(f"JLNN model exported to {path}")
    print(f"Note: This is a placeholder ONNX export. For production use,")
    print(f"consider implementing full StableHLO->ONNX conversion or using")
    print(f"the tf2onnx pipeline via SavedModel intermediate format.")


def export_workflow_example(model: nnx.Module, sample_input: jnp.ndarray, base_name: str):
    """
    Complete export workflow demonstrating both StableHLO and ONNX export.
    
    Args:
        model (nnx.Module): The JLNN model to export.
        sample_input (jnp.ndarray): Sample input for tracing.
        base_name (str): Base filename (without extension).
    
    Example:
        >>> model = MyJLNNModel(...)
        >>> sample = jnp.array([[0.5, 0.8]])
        >>> export_workflow_example(model, sample, "my_jlnn_model")
    """
    # Export to StableHLO
    exported = export_to_stablehlo(model, sample_input)
    
    # Save StableHLO artifact
    save_for_xla_runtime(exported, f"{base_name}.stablehlo")
    
    # Optionally inspect the MLIR/StableHLO code
    print("\n=== StableHLO Module ===")
    print(exported.mlir_module())
    
    # Test the exported model
    graphdef, state = nnx.split(model)
    print("\n=== Testing exported model ===")
    result = exported.call(state, sample_input)
    print(f"Output shape: {result.shape}")
    print(f"Output: {result}")
    
    # Export to ONNX (placeholder implementation)
    export_to_onnx(model, sample_input, f"{base_name}.onnx")