#!/usr/bin/env python3

# Imports
import jax
from flax import nnx
from typing import Dict, Any
import jax.numpy as jnp

class JLNNEngine:
    """
    Computational and optimization core of the JLNN framework.
    
    This class serves as an orchestrator for running a model in a JAX environment. 
    Its main purpose is to encapsulate low-level operations such 
    as JIT graph compilation and automatic differentiation. 

    In the Just-in-time Logical Neural Network architecture, the Engine plays the role of:
    
    1. **Compiler**: Transforms recursive calls to logical nodes 
    into highly efficient code for GPU/TPU.
    
    2. **Training Manager**: Implements an atomic training step that guarantees 
    that the gradient update and subsequent logical projection (constraints) 
    will occur as one inseparable operation.
    
    3. **Abstraction**: Hides the complexity of state management in NNX from the end user.
    """
    
    def __init__(self, model: nnx.Module):
        """
        Initializes the engine for a specific model instance.

        Args:
            model (nnx.Module): An instance of JLNNModel 
                (or other NNX module) that the engine will serve.
        """
        self.model = model

    @nnx.jit
    def infer(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        It performs high-performance, compiled inference on input data.

        Thanks to the @nnx.jit decorator, an optimized computational graph 
        is created on the first call. Each subsequent call with data 
        of the same shape is then executed in the accelerator's native code.

        Args:
            inputs (Dict[str, jnp.ndarray]): Mapping predicate names 
            to input tensors (including batch dimension).

        Returns:
            jnp.ndarray: Truth intervals [L, U] for the root node of the model.
        """
        return self.model(inputs)

    @nnx.jit
    def train_step(self, inputs: Dict[str, jnp.ndarray], targets: jnp.ndarray, 
                   optimizer: Any, loss_fn: Any) -> jnp.ndarray:
        """
        Performs one atomic learning step (forward, backward, update, project).

        This method is the heart of the training process. Within one JIT block:
        
        1. Calculates the forward pass and the loss function value.
        
        2. Use autograd to determine gradients for all trainable parameters.
        
        3. Passes the gradients to an optimizer (e.g. ProjectedOptimizer), 
        which updates the weights and immediately applies logical constraints.

        Args:
            inputs (Dict[str, jnp.ndarray]): Training data (predicate inputs).
            targets (jnp.ndarray): Target truth intervals (labels).
            optimizer (Any): An instance of an NNX-compatible optimizer
                        (typically ProjectedOptimizer from jlnn.training).
            loss_fn (Any): Function for computing the loss (e.g. total_lnn_loss).

        Returns:
            jnp.ndarray: The value of the loss function for the current step (before the update).
        """
        def compute_loss(model):
            # Internal functions for nnx.value_and_grad
            preds = model(inputs)
            return loss_fn(preds, targets)

        # Calculate loss and gradients in one pass
        loss, grads = nnx.value_and_grad(compute_loss)(self.model)
        
        # Updating model weights and enforcing logical axioms
        optimizer.step(self.model, grads)
        
        return loss