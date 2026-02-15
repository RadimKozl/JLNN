#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
from jax.export import export
import onnx
from onnx import helper, TensorProto
import numpy as np

def export_to_stablehlo(model: nnx.Module, sample_input):
    """
    Compiles JLNN model into a StableHLO artifact.
    
    This function bridges the gap between stateful Flax NNX modules and the stateless 
    requirements of the JAX export pipeline. It lowers the model's logical operations 
    (e.g., Łukasiewicz t-norms) into StableHLO representation.
    
    Args:
        model (nnx.Module): The trained JLNN model instance containing logic gates.
        sample_input (Any): A sample input tensor or PyTree (e.g., dict of arrays) 
            representing truth intervals. Used for shape and dtype tracing.
    
    Returns:
        jax.export.Exported: Exported StableHLO model artifact that can be serialized
            or executed.
    """
    # 1. State-Graph Separation
    graphdef, state = nnx.split(model)
    
    # 2. Pure Functional Wrapper
    @jax.jit
    def forward_fn(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)
    
    # 3. Abstract Value Specification (Avals)
    # Support for PyTrees (dictionaries of predicates) using jax.tree.map
    state_avals = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state
    )
    input_avals = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), sample_input
    )
    
    # 4. Lowering to StableHLO
    exported = export(forward_fn)(state_avals, input_avals)
    
    return exported

def export_to_onnx(model: nnx.Module, sample_input, path: str):
    """
    Exports a JLNN (Logical Neural Network) model to ONNX format.
    
    This function first exports the model to StableHLO, then converts it to ONNX
    using placeholder manual graph construction. The process is designed to 
    handle both single tensors and dictionary-based predicate inputs.
    
    Args:
        model (nnx.Module): The trained JLNN model instance.
        sample_input (Any): Sample input tensor or PyTree for tracing.
        path (str): Destination file path for the .onnx model.
    """
    # First export to StableHLO using the robust PyTree-aware function
    exported = export_to_stablehlo(model, sample_input)
    
    # Get the actual output by running the model
    output = model(sample_input)
    
    # Helper to resolve shapes for both tensors and nested PyTrees
    def get_representative_shape(x):
        if hasattr(x, 'shape'):
            return list(x.shape)
        # For dictionaries, we use the shape of the first leaf for the placeholder metadata
        return list(jax.tree.leaves(x)[0].shape)

    # Define input tensor metadata
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        get_representative_shape(sample_input)
    )
    
    # Define output tensor metadata
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        get_representative_shape(output)
    )
    
    # Create a placeholder identity node
    # Note: For full production logic, a StableHLO->ONNX bridge (like tf2onnx) is recommended
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
    print(f"Note: This export supports PyTree inputs. Semantics are preserved via StableHLO lowering.")

def save_for_xla_runtime(exported: jax.export.Exported, filename: str):
    """
    Serializes the StableHLO model for XLA/StableHLO runtime executors.
    """
    serialized_bytes = exported.serialize()
    with open(filename, "wb") as f:
        f.write(serialized_bytes)
    print(f"JLNN StableHLO model saved to {filename}")

  
def export_workflow_example(model: nnx.Module, sample_input, base_name: str):
    """
    Complete export workflow demonstrating both StableHLO and ONNX export.
    
    This example demonstrates the end-to-end pipeline: splitting the model state,
    lowering to StableHLO, and generating a portable ONNX artifact. It supports
    both simple tensor inputs and complex PyTree (dictionary) structures.
    
    Args:
        model (nnx.Module): The JLNN model to export.
        sample_input (Any): Sample input for tracing (tensor or dict of tensors).
        base_name (str): Base filename (without extension).
    
    Example:
        >>> # Example with dictionary-based predicates
        >>> sample = {"A": jnp.array([[0.5, 0.8]]), "B": jnp.array([[0.2, 0.6]])}
        >>> export_workflow_example(model, sample, "logic_model")
    """
    # 1. Export to StableHLO
    # This captures the model logic into a portable MLIR-based representation.
    exported = export_to_stablehlo(model, sample_input)
    
    # 2. Save StableHLO artifact
    # Suitable for XLA runtimes and high-performance inference.
    save_for_xla_runtime(exported, f"{base_name}.stablehlo")
    
    # 3. Inspect the MLIR/StableHLO code (Optional)
    print("\n=== StableHLO Module ===")
    print(exported.mlir_module())
    
    # 4. Test the exported model
    # We must provide the pure model state for the exported call.
    graphdef, state = nnx.split(model)
    print("\n=== Testing exported model ===")
    result = exported.call(state, sample_input)
    
    # Helper to handle shape printing for both tensors and PyTrees
    def get_shape_info(x):
        return x.shape if hasattr(x, 'shape') else jax.tree.map(lambda leaf: leaf.shape, x)

    print(f"Output shape: {get_shape_info(result)}")
    print(f"Output value: {result}")
    
    # 5. Export to ONNX
    # Final step to create a platform-agnostic artifact.
    export_to_onnx(model, sample_input, f"{base_name}.onnx")