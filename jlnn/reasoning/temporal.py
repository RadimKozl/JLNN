#!/usr/bin/env python3

# Imports
from typing import Dict
import jax.numpy as jnp
from jlnn.symbolic.compiler import Node

class AlwaysNode(Node):
    """
    Temporal operator G (Globally / Always) for time series.

    In the context of temporal logic, the formula G(A) is true at time t 
    if and only if the sub-formula A is true at all time steps (future and present). 
    
    In JLNN, this operation is implemented as a generalized conjunction (AND) 
    over the time axis. It uses Gödel's t-norm (minimum), 
    which means that the resulting truth value of the interval 
    is bounded by the least true moment in a given sequential window.

    This node is key for detecting persistent states or invariants.
    """
    
    def __init__(self, child: Node, window_size: int = None):
        """
        Initializes the 'Always' node.

        Args:
            child (Node): A subtree (formula) whose truth we monitor over time.
            window_size (int, optional): The size of the sliding window. 
                If None, the operator is applied to the entire length of the input sequence.
        """
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the truth interval for the Globally operator.

        The input data 'values' is expected to be in the form (batch, time, features). 
        The node first recursively obtains the truth values 
        from the child and then performs aggregation over the timeline.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary of input tensors.

        Returns:
            jnp.ndarray: Aggregated interval [L, U] of shape (batch, 2).
        """
        # Get the truth of the child: shape [batch, time, 2]
        a = self.child.forward(values) 
        
        # Aggregate over time axis (axis=1) using t-norm min.
        # The result represents the 'worst possible scenario' of truth over time.
        return jnp.min(a, axis=1)

class EventuallyNode(Node):
    """
    Temporal operator F (Eventually / Finally) for time series.

    The formula F(A) is true if there is at least one point in time 
    (now or in the future) at which the sub-formula A is true.
    
    In JLNN, this operator is implemented as a generalized disjunction (OR) 
    over a time axis (t-conorma). It uses maximum, which means that the overall truth 
    is determined by the 'best moment' of the entire sequence.

    Useful for detecting events that must occur but 
    are not specified exactly when (e.g., response to a stimulus).
    """

    def __init__(self, child: Node, window_size: int = None):
        """
        Initializes the 'Finally' node.

        Args:
            child (Node): A formula that we expect to occur in time.
            window_size (int, optional): The range of the operator in time.
        """
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the truth interval for the Eventually operator.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary of input tensors.

        Returns:
            jnp.ndarray: Aggregated interval [L, U] of shape (batch, 2).
        """
        # Get the truth of the child: shape [batch, time, 2]
        a = self.child.forward(values)
        
        # Aggregate over time using t-conorm max.
        # The result represents 'existential' truth in time.    
        return jnp.max(a, axis=1)