#!/usr/bin/env python3

# Imports
import optax
from flax import nnx
from typing import Any
from jlnn.nn.constraints import apply_constraints

class ProjectedOptimizer:
    """
    Optimizer with support for logical constraints (Constraint Projection).
    
    This class wraps the standard optax optimizer 
    and ensures that after each learning step the model 
    parameters satisfy logical axioms (e.g. w >= 1 for LNN gates).
    """

    def __init__(self, optimizer: optax.GradientTransformation, model: nnx.Module):
        """
        Initializes the optimizer state for the given model.

        Args:
            optimizer: Optax transformation chain (e.g. optax.adam(1e-3)).
            model: JLNNModel instance whose parameters we will optimize.
        """
        self.optimizer = optimizer
        # Initialize the optimizer state for trainable parameters only
        self.opt_state = optimizer.init(nnx.state(model, nnx.Param))

    def step(self, model: nnx.Module, grads: Any):
        """
        It performs one optimization step followed by projection onto a logical set.

        Args:
            model: The model whose parameters we are updating.
            grads: Gradients calculated using jax.grad or jax.value_and_grad.
        """
        # 1. Getting the current status of parameters
        params = nnx.state(model, nnx.Param)
        
        # 2. Calculating updates (Adam/SGD mechanics)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        
        # 3. Applying updates to parameters
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        
        # 4. PROJECTION: Enforcing logical constraints (e.g. trimming weights)
        # This transforms standard SGD into Projected Gradient Descent.
        apply_constraints(model)