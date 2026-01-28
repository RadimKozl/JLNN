#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx

def clip_weights(model: nnx.Module):
    """
    Ensures that all trainable weights in logic gates are >= 1.0.

    In Logical Neural Networks (LNN) and Łukasiewicz logic, 
    strict adherence to the condition w >= 1.0 is necessary 
    to preserve the semantic interpretability of t-norms and t-conorms. 
    If the weights were to fall below this limit, 
    the gates would lose their logical identity and start behaving like standard neurons, 
    making it impossible to translate the network back into a set of symbolic rules.

    This function implements the 'Projected Gradient Descent' method. 
    After each parameter update (when the gradient may shift the weights into an invalid space) 
    it projects the weights back to the allowed region [1.0, inf).

    The function automatically detects:
    - 'weights': Collective weights for AND, OR, NAND, NOR, XOR gates.
    - 'weight': Individual weight for unary operations like NOT.

    Args:
        model (nnx.Module): An instance of the Flax NNX model. 
            The function iterates through the entire parameter graph 
            and modifies the nnx.Param values ​​in-place.
    """
    # We traverse the model parameter graph using the NNX iterator
    for path, param in model.iter_graph():
        # Check if this is a trainable parameter
        if isinstance(param, nnx.Param):
            # We target weights across all gate types in gates.py
            # path[-1] represents the attribute name in the class definition
            if any(name in path[-1] for name in ('weights', 'weight')):
                # Applying the maximum operation to trim the lower bound.
                # In JAX, this operation is efficient and preserves the integrity of the calculation.
                param.value = jnp.maximum(1.0, param.value)


def clip_predicates(model: nnx.Module):
    """
    Ensures logical integrity and consistency of LearnedPredicate parameters.

    This function enforces an axiomatic relationship between the lower (L) and upper (U) 
    bounds of the truth interval directly at the level of the ramp_sigmoid activation parameters. 
    In the JLNN framework, the lower bound represents the confirmed truth 
    and the upper bound represents the maximum possible truth; therefore, by definition, L <= U must hold.

    In the function used f(x) = clip(slope * (x - offset) + 0.5, 0, 1), 
    the 'offset' parameter plays the role of a decision boundary on the X-axis. 
    A smaller offset value means that the function starts to rise earlier (at lower input values). 
    In order to maintain the integrity of the interval, 
    the upper bound (U) must always "cover" the lower bound (L), 
    which mathematically requires the condition: offset_u <= offset_l.

    Mechanism of operation:
    1. The function searches for all 'offset_u' parameters in the model graph.
    2. Dynamically searches for the corresponding 'offset_l' within the same module.
    3. If the gradient moves 'offset_u' above the 'offset_l' value, 
    the function moves (projects) it back to the equality boundary.

    Args:
        model (nnx.Module): An instance of the Flax NNX model for which input predicate consistency is required. 
            Modification is done in-place.
    """
    # We traverse the model graph and look for parameters affecting the upper bounds of the predicates
    for path, param in model.iter_graph():
        if isinstance(param, nnx.Param):
            # We are specifically targeting parameters defined in jlnn.nn.predicates
            if path[-1] == 'offset_u':
                # We access the parent module through the path in the graph
                parent_path = path[:-1]
                try:
                    # Finding the sibling parameter offset_l
                    offset_l_path = parent_path + ('offset_l',)
                    offset_l_value = model.get_at(offset_l_path).value
                    
                    # Projection of the parameter offset_u into the allowed half-space.
                    # We use jnp.minimum because for an earlier onset of U
                    # we need an offset smaller than or equal to L.
                    param.value = jnp.minimum(offset_l_value, param.value)
                    
                except (AttributeError, KeyError):
                    # Handling the case where a parameter exists outside the expected structure
                    pass


def apply_constraints(model: nnx.Module):
    """
    Aggregates and applies all logical and structural constraints to the model parameters.

    This function represents a critical point in the JLNN training loop, 
    implementing a method called 'Projected Gradient Descent'. 
    It must be called immediately after each parameter update by an optimizer (e.g. Optax), 
    but before the next forward pass.

    Main tasks of the function:
    1. Guarantees interpretability: Returns gate weights to the range >= 1.0,
    ensuring that gates behave as t-norms/t-conorms.
    
    2. Prevents logical conflicts: Adjusts predicate parameters so that L <= U always holds, 
    thus eliminating the emergence of internal contradictions at the input data level.
    
    3. Stabilizes learning: Keeps the model in the space of valid logical formulas, 
    which prevents numerical instability in deep logical graphs.

    Args:
        model (nnx.Module): An instance of 
            a Flax NNX model to which the constraints apply throughout the parameter tree.
    """
    # 1. Treatment of weights for all gates (AND, OR, NAND, NOR, XOR, NOT, Implication)
    # Ensures that w >= 1.0 to preserve operator interpretability.
    clip_weights(model)
    
    # 2. Treatment of predicates for interval stability (grounding consistency)
    # Ensures that offset_u <= offset_l so that the lower bound does not exceed the upper bound.
    clip_predicates(model)