#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn import gates, predicates, constraints

def test_apply_constraints_to_gate(rngs):
    """
    Verifies the projection of gate weights to the valid logical domain.

    In Logical Neural Networks, weights must remain >= 1.0 to preserve the
    semantics of the underlying t-norms. This test manually sets a weight 
    below this threshold (0.2) and verifies that apply_constraints 
    correctly projects it back to 1.0 while leaving valid weights (1.5) 
    untouched.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    gate = gates.WeightedOr(num_inputs=2, rngs=rngs)
    
    # Manually violate the constraint for the first weight
    # Using [...] ensures direct Array assignment in Flax NNX
    gate.weights[...] = jnp.array([0.2, 1.5])
    
    # Apply the Projected Gradient Descent step
    constraints.apply_constraints(gate)
    
    # Verify that the violating weight was clipped and the valid one preserved
    assert gate.weights[...][0] == 1.0
    assert gate.weights[...][1] == 1.5

def test_apply_constraints_to_predicate(rngs):
    """
    Ensures that predicate offsets maintain logical interval consistency (L <= U).

    Predicates map numeric data to truth intervals. If the upper bound offset 
    exceeds the lower bound offset, it can create a logical contradiction 
    where the lower bound is higher than the upper bound. This test verifies 
    that apply_constraints enforces offset_u <= offset_l.

    Args:
        rngs (nnx.Rngs): Flax NNX random number generator collection.
    """
    pred = predicates.LearnedPredicate(in_features=1, rngs=rngs)
    
    # Set offsets such that they would cause L > U in certain input ranges
    pred.offset_l[...] = jnp.array([0.0])
    pred.offset_u[...] = jnp.array([0.5]) 
    
    # Apply the logical consistency constraint
    constraints.apply_constraints(pred)
    
    # Verify that the contradiction was resolved by clipping offset_u
    assert pred.offset_u[...] <= pred.offset_l[...]