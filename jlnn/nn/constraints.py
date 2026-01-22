#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx

def clip_weights(model: nnx.Module):
    """
    Ensures that all trainable weights in logic gates are >= 1.0.

    In Logical Neural Networks (LNN) and Łukasiewicz logic, the condition w >= 1 
    is necessary to maintain logical interpretability. If the weights were to fall below 1.0, 
    the gate would start to behave more like a classical neuron in a perceptron 
    and would lose the t-norm/t-conorm properties.

    This function finds all weights across the model graph, 
    including embedded gates (such as WeightedNand, WeightedNor, 
    or WeightedXor), and trims them back to the allowed space.

    Args:
        model (nnx.Module): The Flax NNX model instance whose parameters will be checked and edited.
        
    """
    # We traverse the model parameter graph using the Flax NNX iterator
    for path, param in model.iter_graph():
        # Check if it is a trainable parameter
        if isinstance(param, nnx.Param):
            # We target parameters named 'weights' across all gates
            if path[-1] == 'weights':
                # We perform a cutoff at a lower bound of 1.0 to preserve t-norm semantics
                param.value = jnp.maximum(1.0, param.value)


def clip_predicates(model: nnx.Module):
    """
    Ensures the logical integrity of LearnedPredicate predicate parameters.

    This function enforces bindings between the lower (L) and upper (U) bound parameters. 
    The goal is that the bias and slope for the upper bound 
    are always set to allow for a wider or equal truth space than the lower bound parameters.

    In this version, we focus on bias, where we ensure that the bias for the upper bound (bias_u) 
    is always greater than or equal to the bias for the lower bound (bias_l), 
    which naturally maintains the relationship L <= U and prevents a logical conflict.

    Args:
        model (nnx.Module): Model containing LearnedPredicate layers.
    """
    for path, param in model.iter_graph():
        if isinstance(param, nnx.Param):
            # We are looking for the bias upper bound
            if path[-1] == 'bias_u':
                # Get the path to the corresponding bias_l in the same module
                parent_path = path[:-1]
                try:
                    bias_l_path = parent_path + ('bias_l',)
                    bias_l_value = model.get_at(bias_l_path).value
                    
                    # Force bias_u >= bias_l
                    param.value = jnp.maximum(bias_l_value, param.value)
                except (AttributeError, KeyError):
                    pass


def apply_constraints(model: nnx.Module):
    """
    It aggregates and applies all logical and structural constraints to the model parameters.

    This function is a critical point in the JLNN training loop. 
    It must be called immediately after each parameter update by the optimizer (e.g. Optax), 
    but before the next forward pass.

    This mechanism is implemented by the so-called 'Projected Gradient Descent', 
    where we constantly return the model to the space of valid logical axioms, 
    thereby ensuring the interpretability of gates and preventing logical conflicts.

    Args:
        model (nnx.Module): The Flax NNX model instance to which the constraints apply.
    """
    # 1. Treatment of weights for all gates (AND, OR, NAND, NOR, XOR, Implication)
    clip_weights(model)
    # 2. Treatment of predicates for interval stability (L <= U)
    clip_predicates(model)