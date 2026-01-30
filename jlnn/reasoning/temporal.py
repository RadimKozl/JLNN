#!/usr/bin/env python3

# Imports
from typing import Dict
import jax.numpy as jnp
from jlnn.symbolic.compiler import Node

class AlwaysNode(Node):
    r"""
    Implementation of the 'Always' (Globally) temporal operator, denoted as $\mathcal{G}$.

    In Linear Temporal Logic (LTL), the formula $\mathcal{G}\phi$ is true if the 
    sub-formula $\phi$ holds at every time step within a given sequence. 
    Within the JLNN framework, this is realized as a generalized conjunction (AND) 
    over the temporal axis. 

    It utilizes the Gödel t-norm (minimum) to aggregate truth intervals, ensuring 
    that the resulting lower bound represents the "least true" moment in the 
    time series.

    Attributes:
        child (Node): The logical subtree or formula to be evaluated over time.
        window_size (Optional[int]): The specific temporal look-ahead window. 
            If None, the operator applies to the entire input sequence.
    """
    
    def __init__(self, child: Node, window_size: int = None):
        """
        Initializes the Always (G) node.

        Args:
            child: The sub-formula to monitor.
            window_size: The temporal range for the operator.
        """
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the minimum truth interval across the temporal dimension.

        Computes the intersection of truth values over time, effectively 
        finding the invariant truth level of the sequence.

        Args:
            values: A dictionary of input tensors where the temporal dimension 
                is expected at axis 1.

        Returns:
            A JAX array of truth intervals $[L, U]$ with the temporal 
            dimension collapsed via the `min` operation.
        """
        # Obtain child activations: shape (batch, time, 2)
        a = self.child.forward(values)
        # Aggregate across the time axis (axis=1)
        return jnp.min(a, axis=1)

class EventuallyNode(Node):
    r"""
    Implementation of the 'Eventually' (Finally) temporal operator, denoted as $\mathcal{F}$.

    In LTL, the formula $\mathcal{F}\phi$ is true if the sub-formula $\phi$ holds at 
    least once at some point in the future or present. In JLNN, this is 
    implemented as a generalized disjunction (OR) over the temporal axis.

    It utilizes a t-conorm (maximum) to aggregate truth intervals, meaning the 
    overall truth is determined by the "most true" moment in the sequence.

    Attributes:
        child (Node): The logical subtree to be evaluated.
        window_size (Optional[int]): The specific temporal look-ahead window.
    """

    def __init__(self, child: Node, window_size: int = None):
        """
        Initializes the Eventually (F) node.

        Args:
            child: The sub-formula expected to occur.
            window_size: The temporal range for the operator.
        """
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the maximum truth interval across the temporal dimension.

        Determines the peak truth value within the sequence, identifying if the 
        condition is met at any point.

        Args:
            values: A dictionary of input tensors.

        Returns:
            A JAX array of truth intervals $[L, U]$ with the temporal 
            dimension collapsed via the `max` operation.
        """
        # Obtain child activations: shape (batch, time, 2)
        a = self.child.forward(values)
        # Aggregate across the time axis (axis=1)
        return jnp.max(a, axis=1)