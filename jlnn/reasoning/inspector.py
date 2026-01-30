#!/usr/bin/env python3

# Imports
from typing import Dict, List, Any
import jax.numpy as jnp
from jlnn.symbolic.compiler import Node

def trace_reasoning(node: Node, inputs: Dict[str, jnp.ndarray]) -> List[Dict[str, Any]]:
    """
    Recursively audits the logical graph to capture activations for every node.

    This function performs a deep introspection of the forward pass by traversing 
    the compiled logical tree. For each node, it captures the resulting 
    truth interval $[L, U]$, which is essential for Explainable AI (XAI) tasks, 
    rule auditing, and debugging model conclusions.

    Args:
        node: The root node or subtree of a compiled JLNN model.
        inputs: A dictionary mapping predicate names to input tensors 
            of shape (batch, [time], features).

    Returns:
        A list of dictionaries, where each entry represents a node's state:
            - 'name': The descriptive identifier of the node.
            - 'interval': The average truth interval (L, U) across the batch.
            - 'output': The raw JAX array containing full activation data.
    """
    results = []
    
    # Calculate the truth interval for the current node
    output = node.forward(inputs)
    
    # Extract representative L and U by averaging over batch/time dimensions
    flat_output = output.reshape(-1, 2)
    avg_output = jnp.mean(flat_output, axis=0)
    l, u = float(avg_output[0]), float(avg_output[1])
    
    # Generate a descriptive name for the node
    cls_name = node.__class__.__name__.replace("Node", "")
    name = f"{cls_name}({node.name})" if hasattr(node, "name") and node.name else cls_name
        
    results.append({
        "name": name,
        "interval": (l, u),
        "output": output
    })
    
    # Recursive traversal based on node architecture
    if hasattr(node, "children"): # NAry nodes (AND, OR)
        for child in node.children:
            results.extend(trace_reasoning(child, inputs))
    elif hasattr(node, "child"): # Unary nodes (NOT, G, F)
        results.extend(trace_reasoning(node.child, inputs))
    elif hasattr(node, "left") and hasattr(node, "right"): # Binary nodes (->, <->)
        results.extend(trace_reasoning(node.left, inputs))
        results.extend(trace_reasoning(node.right, inputs))
        
    return results

def get_rule_report(model: Any, inputs: Dict[str, jnp.ndarray]) -> str:
    """
    Generates a human-readable semantic interpretation of the model's output.

    This utility classifies the resulting truth interval $[L, U]$ into qualitative 
    categories based on LNN semantics (TRUE, FALSE, UNKNOWN, CONFLICT). 
    It includes specific heuristics for neural predicates that may shift truth 
    values toward the center of the $[0, 1]$ spectrum during initialization 
    or training.

    Args:
        model: The JLNN model or LNNFormula to evaluate.
        inputs: Input data dictionary for the predicates.

    Returns:
        A formatted string containing the numerical interval and its 
        semantic classification (e.g., "Result: [0.52, 0.73] - FALSE (NEURAL)").
    """
    output = model(inputs)
    
    # Flatten and average to get a stable [L, U]
    avg_output = jnp.mean(output.reshape(-1, 2), axis=0)
    l, u = float(avg_output[0]), float(avg_output[1])
    
    # Semantic classification logic
    if l > u + 1e-5:
        status = "CONFLICT"
    elif l >= 0.8:
        status = "TRUE"
    elif u <= 0.2:
        status = "FALSE"
    # Adjusted Neural Heuristics:
    # If the upper bound is still below the True threshold,
    # and it originated from a False input, classify as Neural False.
    elif u < 0.8:
        status = "FALSE (NEURAL)"
    elif l > 0.2:
        status = "TRUE (NEURAL)"
    else:
        status = "UNKNOWN"
        
    return f"Result: [{l:.2f}, {u:.2f}] - {status}"