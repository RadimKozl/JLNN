#!/usr/bin/env python3
from typing import Dict, List, Any
import jax.numpy as jnp
from jlnn.symbolic.compiler import Node

def trace_reasoning(node: Node, inputs: Dict[str, jnp.ndarray]) -> List[Dict[str, Any]]:
    """
    Recursively audits the logical graph to capture activations for every node.
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
    Generates a human-readable interpretation of the model's output.
    
    Refined to handle neural predicates that shift 'False' inputs 
    toward the center (e.g., [0.52, 0.73]).
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