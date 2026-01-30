#!/usr/bin/env python3

# Imports
import jax
import optax
from flax import nnx
import jax.numpy as jnp
from jlnn.training.optimizers import ProjectedOptimizer
from jlnn.symbolic.compiler import LNNFormula

def test_projected_optimizer_constraints(rngs):
    r"""
    Ensures that the ProjectedOptimizer enforces logical constraints after a step.

    In Logical Neural Networks (LNNs), weights ($w$) of conjunctions and 
    disjunctions must be maintained $\ge 1$ to preserve the logical 
    bounds and avoid collapsing into a standard neural network.
    
    This test:
    1. Initializes a model with a logical AND gate.
    2. Performs an optimization step with large positive gradients.
    3. Verifies that the subsequent projection step forces weights 
       back to the valid range [1, inf) using JAX PyTree traversal.
    """
    model = LNNFormula("A & B", rngs)
    
    # Standard Adam optimizer wrapped in our ProjectedOptimizer
    base_opt = optax.adam(1e-1)
    optimizer = ProjectedOptimizer(base_opt, model)
    
    # Extract current parameters (nnx.State is a PyTree)
    params = nnx.state(model, nnx.Param)
    
    # Create artificial gradients that push weights DOWN.
    # Updates: params - lr * grads -> large positive grads = drastic decrease.
    fake_grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 100.0, params)
    
    # Execute the atomic step (Update + Projection)
    optimizer.step(model, fake_grads)
    
    # Retrieve updated state
    new_params = nnx.state(model, nnx.Param)
    
    # Use JAX tree_util to flatten the state. This is the most robust way 
    # as it doesn't depend on NNX-specific methods like .flat() or .dict()
    flat_params, _ = jax.tree_util.tree_flatten_with_path(new_params)
    
    found_weights = False
    for path, value in flat_params:
        # path is a tuple of DictKey/GetAttr objects. Convert to string to check.
        path_str = str(path).lower()
        
        if "weights" in path_str:
            found_weights = True
            # In JAX flatten, 'value' is already the leaf (the JAX array)
            assert jnp.all(value >= 1.0 - 1e-5), (
                f"Constraint violation at {path_str}: {value} is < 1.0. "
                "ProjectedOptimizer failed to enforce axioms."
            )
    
    assert found_weights, "Test sanity check failed: No 'weights' parameters were found in the model."

def test_optimizer_state_initialization(rngs):
    """
    Verifies that the optimizer correctly initializes its internal state.
    
    Checks if the Optax transformation is properly wrapped and the 
    initial state of the optimizer is created for the model's parameters.
    """
    model = LNNFormula("A", rngs)
    optimizer = ProjectedOptimizer(optax.sgd(1e-2), model)
    
    assert optimizer.opt_state is not None
    assert hasattr(optimizer, "optimizer"), "ProjectedOptimizer must expose the base optimizer"