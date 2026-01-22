#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx


def clip_weights(model: nnx.Module):
    """
    Ensures that all trainable weights in logic gates are >= 1.0.
    
    In Logical Neural Networks (LNN) and Łukasiewicz logic, 
    the condition w >= 1 is necessary to maintain logical interpretability. 
    If the weights were to fall below 1.0, the gate would start to behave more like 
    a classical neuron in a perceptron and would lose the t-norm/t-conorm properties.
    
    This function implements Projected Gradient Descent by "clipping" 
    the weights back into the allowed space after each optimizer step.

    Args:
        model (nnx.Module): The Flax NNX model instance whose parameters will be checked and edited.
    """    
    # We traverse the model parameter graph using the Flax NNX iterator
    for path, param in model.iter_graph():
        # Check if it is a trainable parameter
        if isinstance(param, nnx.Param):
            # We are specifically targeting parameters named 'weights'
            # in the WeightedOr, WeightedAnd, and WeightedImplication gates
            if path[-1] == 'weights':
                # We perform the clipping at the lower limit of 1.0
                param.value = jnp.maximum(1.0, param.value)


def clip_predicates(model: nnx.Module):
    """
    Ensures that Learned Predicate parameters maintain logical integrity.

    This function passes predicate parameters 
    and can enforce bindings between the lower (L) and upper (U) bound parameters. 
    The goal is that the bias and slope for the upper (U) bound are always set to allow 
    for a wider or equal truth space than the lower (L) bound parameters.

    In this version, we focus on bias, 
    where we ensure that the bias for the upper bound 
    is always greater than or equal to the bias for the lower bound (for the same steepness), 
    which naturally maintains the relationship L <= U.

    Args:
        model (nnx.Module): Model containing LearnedPredicate layers.
    """
    for path, param in model.iter_graph():
        # We look for specific modules of type LearnedPredicate (if they are named that way)
        # or we identify the parameters bias_u and bias_l
        if isinstance(param, nnx.Param):
            # We aim for the bias upper bound
            if path[-1] == 'bias_u':
                # Get the path to the corresponding bias_l (lower bound)
                # path is a tuple, e.g. ('node', 'predicate_1', 'bias_u')
                parent_path = path[:-1]
                try:
                    # We will try to find bias_l in the same module
                    bias_l_path = parent_path + ('bias_l',)
                    bias_l_value = model.get_at(bias_l_path).value
                    
                    # Force bias_u >= bias_l
                    # This ensures that sigmoid(x + bias_u) >= sigmoid(x + bias_l)
                    param.value = jnp.maximum(bias_l_value, param.value)
                except (AttributeError, KeyError):
                    # If bias_l does not exist in this context, we skip
                    pass


def apply_constraints(model: nnx.Module):
    """
    It aggregates and applies all logical and structural constraints to the model parameters.

    This function is a critical point in the JLNN training loop. 
    It must be called immediately after each parameter update by the optimizer (e.g. Optax), 
    but before the next forward pass.

    Without applying these constraints, the model during optimization could:
    1. Lose the interpretability of the gates (if the weights fall below 1.0).
    2. Generate logical contradictions (if the lower limit L exceeds the upper limit U).

    This mechanism is implemented by the so-called 'Projected Gradient Descent', 
    where we constantly return the model to the space of valid logical axioms.

    Args:
        model (nnx.Module): The Flax NNX model instance to which the constraints apply.
    """
    # 1. Enforcing gate weights (w >= 1.0) to preserve t-norm semantics
    clip_weights(model)
    # 2. Enforcing predicate consistency (L <= U) to prevent contention
    clip_predicates(model)