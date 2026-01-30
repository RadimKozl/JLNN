#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx

def clip_weights(model: nnx.Module):
    """
    Ensures that all trainable weights in logic gates satisfy the condition w >= 1.0.

    In Logical Neural Networks (LNN) using Łukasiewicz semantics, maintaining 
    weights >= 1.0 is crucial for the interpretability of t-norms and t-conorms. 
    If weights fall below this threshold, gates lose their identity as logical 
    operators and behave like standard neural nodes.

    This function implements a 'Projected Gradient Descent' step by projecting 
    violating weights back to the valid domain [1.0, inf).

    Args:
        model (nnx.Module): The Flax NNX model/module to be constrained. 
            The function traverses the entire graph and updates parameters in-place.
    """
    # We iterate through the model graph to find all Param nodes
    for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Param):
            param_name = str(path[-1])
            # Target collective weights (AND, OR, XOR, etc.) and unary weights (NOT)
            if 'weight' in param_name:
                # Use [...] for both reading the value and in-place assignment 
                # to avoid DeprecationWarnings and ensure JAX compatibility.
                node[...] = jnp.maximum(1.0, node[...])

def clip_predicates(model: nnx.Module):
    """
    Ensures logical consistency in grounding layers (predicates) by maintaining L <= U.

    For a LearnedPredicate, the lower bound (L) must never exceed the upper bound (U).
    This is achieved by adjusting the offset parameters such that the transition 
    for the upper bound does not lag behind the lower bound.

    Args:
        model (nnx.Module): The Flax NNX model containing LearnedPredicate modules.
    """
    for path, module in nnx.iter_graph(model):
        # Check for modules that have both lower and upper offsets (Predicates)
        if hasattr(module, 'offset_l') and hasattr(module, 'offset_u'):
            # Enforce the constraint: offset_u must be <= offset_l.
            # This ensures that for any input x, the resulting interval [L, U] is valid.
            module.offset_u[...] = jnp.minimum(module.offset_u[...], module.offset_l[...])

def apply_constraints(model: nnx.Module):
    """
    Top-level function to aggregate and apply all logical and structural constraints.

    This should be called immediately after the optimizer's update step 
    but before the next forward pass. It keeps the model within the 
    feasible space of logical formulas.

    Args:
        model (nnx.Module): The Flax NNX model to be constrained.
    """
    # 1. Constrain gate weights to preserve logical operator semantics
    clip_weights(model)
    
    # 2. Constrain predicate offsets to ensure interval consistency (no contradictions)
    clip_predicates(model)