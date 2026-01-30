#!/usr/bin/env python3
import jax
from flax import nnx
from typing import Dict, Any, Callable
import jax.numpy as jnp

class JLNNEngine(nnx.Module):
    """
    The orchestration engine for JLNN models, handling JIT compilation and training logic.

    This class wraps a logical model to provide optimized inference and training 
    steps. By inheriting from nnx.Module, it ensures that JAX can correctly 
    trace the model state during compilation.
    """

    def __init__(self, model: nnx.Module):
        """
        Initializes the engine with a specific logical model.

        Args:
            model (nnx.Module): The compiled LNN formula or neural logic network.
        """
        self.model = model

    @nnx.jit
    def infer(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Executes a JIT-compiled forward pass.

        Args:
            inputs (Dict[str, jnp.ndarray]): Dictionary mapping predicate names 
                to input arrays of shape (batch, time/feature).

        Returns:
            jnp.ndarray: Truth intervals of shape (batch, time/feature, 2).
        """
        return self.model(inputs)

    @nnx.jit
    def train_step(self, inputs: Dict[str, jnp.ndarray], targets: jnp.ndarray, 
                   optimizer: nnx.Optimizer, loss_fn: Callable) -> jnp.ndarray:
        """
        Performs an atomic training step: Forward -> Loss -> Backward -> Update.

        Args:
            inputs: Model input features.
            targets: Target truth intervals.
            optimizer: Flax NNX optimizer instance.
            loss_fn: Scalar-valued loss function.

        Returns:
            jnp.ndarray: The loss value for the current step.
        """
        def compute_loss(model):
            preds = model(inputs)
            return loss_fn(preds, targets)

        loss, grads = nnx.value_and_grad(compute_loss)(self.model)
        optimizer.step(self.model, grads)
        return loss