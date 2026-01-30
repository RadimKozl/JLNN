#!/usr/bin/env python3

# Imports
import jax
from flax import nnx
from typing import Dict, Any, Callable
import jax.numpy as jnp

class JLNNEngine(nnx.Module):
    """
    Orchestration engine for JLNN models, managing high-performance execution.

    This class serves as a high-level wrapper for compiled logical formulas,
    facilitating JIT-compiled inference and atomic training operations. 
    By extending `nnx.Module`, it leverages the Flax NNX state management 
    system, allowing JAX to efficiently trace and optimize the model's 
    computational graph across various hardware accelerators (CPU, GPU, TPU).

    Attributes:
        model (nnx.Module): The compiled logical network or formula to be executed.
    """

    def __init__(self, model: nnx.Module):
        """
        Initializes the engine with a target logical model.

        Args:
            model: A compiled LNN formula or neural logic network instance 
                conforming to the NNX module interface.
        """
        self.model = model

    @nnx.jit
    def infer(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Executes a JIT-compiled forward pass through the logical graph.

        Args:
            inputs: A dictionary mapping predicate names (strings) to input 
                tensors of shape (batch, [time], features). The engine handles 
                multi-dimensional data for both static and temporal reasoning.

        Returns:
            A JAX array of truth intervals [L, U] with shape (batch, [time], 2).
        """
        return self.model(inputs)

    @nnx.jit
    def train_step(self, 
                   inputs: Dict[str, jnp.ndarray],
                   targets: jnp.ndarray,
                   optimizer: nnx.Optimizer,
                   loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        """
        Performs an atomic training step: Forward, Backward, and State Update.

        This method encapsulates the complete optimization cycle within a single 
        JIT-compiled block. It computes the loss based on target intervals, 
        calculates gradients via automatic differentiation, and updates the 
        model parameters using the provided optimizer.

        Args:
            inputs: Input data dictionary for the forward pass.
            targets: Ground truth intervals [L, U] representing the desired 
                logical state for the output nodes.
            optimizer: A Flax NNX optimizer instance (e.g., Adam, SGD, 
                or a custom constrained optimizer).
            loss_fn: A callable that computes a scalar loss value from 
                model predictions and target intervals.

        Returns:
            The scalar loss value computed for the current step (pre-update).
        """
        def compute_loss(model):
            preds = model(inputs)
            return loss_fn(preds, targets)

        loss, grads = nnx.value_and_grad(compute_loss)(self.model)
        optimizer.step(self.model, grads)
        return loss