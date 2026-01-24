#!/usr/bin/env python3

# Imports
from typing import Dict, List, Any
import jax.numpy as jnp
from flax import nnx
from jlnn.symbolic.compiler import Node, PredicateNode, BinaryGateNode, NAryGateNode, UnaryGateNode


def trace_reasoning(node: Node, inputs: Dict[str, jnp.ndarray]) -> List[Dict[str, Any]]:
    """
    It recursively traverses the computational graph (tree) and extracts activations for each logical node.

    This function performs an in-depth analysis (instrospection) of the forward pass. 
    For each node in the hierarchy, it captures the resulting truth interval [L, U]. 
    This is essential for auditing the model, tuning the logic rules, 
    and understanding why the model reached a given conclusion.

    Args:
        node (Node): The root node or subtree of the compiled JLNN model. 
        inputs (Dict[str, jnp.ndarray]): A dictionary of input data, 
        where the keys correspond to the names of the predicates.

    Returns:
        List[Dict[str, Any]]: List of activation records. Each record contains:
            - 'name': Node identifier (e.g. predicate name or gate type).
            - 'type': Node class (for structural overview).
            - 'interval': Average truth interval [L, U] calculated over batch.
    """
    results = []
    
    # Calculate the current node (subtree inference)
    output = node.forward(inputs)
    
    # Heuristic determination of a human-readable node name
    if isinstance(node, PredicateNode):
        node_name = f"Predicate({node.name})"
        children = []
    elif isinstance(node, BinaryGateNode):
        node_name = f"BinaryGate({node.gate.__class__.__name__})"
        children = [node.left, node.right]
    elif isinstance(node, NAryGateNode):
        node_name = f"NAryGate({node.gate.__class__.__name__}, fans={len(node.children)})"
        children = node.children
    elif isinstance(node, UnaryGateNode):
        node_name = f"UnaryGate({node.gate.__class__.__name__})"
        children = [node.child]
    else:
        node_name = "UnknownNode"
        children = []

    # Aggregation of results (averaging over a batch to simplify inspection)
    results.append({
        "name": node_name,
        "type": type(node).__name__,
        "interval": jnp.mean(output, axis=0) if output.ndim > 1 else output
    })

    # Recursive Descent DFS (Depth-First Search)
    for child in children:
        results.extend(trace_reasoning(child, inputs))
        
    return results

def print_logic_summary(model: Any, inputs: Dict[str, jnp.ndarray]):
    """
    Generates and prints a readable logical reasoning report to the console.
    
    The output is formatted as a table that displays the hierarchical structure 
    of the calculation from root to leaves. It allows for a quick visual check 
    of the 'thought flow' of the model and identification of 
    nodes with high uncertainty (large difference between L and U).


    Args:
        model: A JLNNModel instance containing the compiled root.
        inputs: Test or production data for inspection.
    """
    print(f"\n{'='*60}")
    print(f"{'JLNN REASONING INSPECTOR':^60}")
    print(f"{'='*60}")
    print(f"{'Logical Node':<35} | {'Lower (L)':<10} | {'Upper (U)':<10}")
    print(f"{'-'*35}-+-{'-'*10}-+-{'-'*10}")

    trace = trace_reasoning(model.root, inputs)
    
    # Iterate through captured activations
    for entry in trace:
        l, u = entry["interval"]
        name = entry["name"]
        print(f"{name:<35} | {l:10.4f} | {u:10.4f}")
    
    print(f"{'='*60}\n")

def get_rule_report(model: Any, inputs: Dict[str, jnp.ndarray]) -> str:
    """
    It creates a brief textual summary of the inference result in human language.

    Based on the resulting interval [L, U], it classifies the system state 
    into categories according to LNN semantics (True, False, Unknown, Conflict).

    Args:
        model: Model to evaluate.
        inputs: Input data.

    Returns:
        str: String containing the interpretation of the result and the numerical interval.
    """
    output = model(inputs)
    l, u = jnp.mean(output, axis=0)
    
    # Decision logic based on expert thresholds
    status = "UNKNOWN"
    if l > 0.8: 
        status = "TRUE (Confident)"
    elif u < 0.2: 
        status = "FALSE (Confident)"
    elif l > 0.4 and u < 0.6: 
        status = "UNCERTAIN / CONTRADICTORY"
    elif u - l > 0.5:
        status = "INSUFFICIENT INFORMATION"
    
    return f"Model conclusion: {status} | Interval: [{l:.3f}, {u:.3f}]"