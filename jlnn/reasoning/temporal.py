#!/usr/bin/env python3
from typing import Dict
import jax.numpy as jnp
from jlnn.symbolic.compiler import Node

class AlwaysNode(Node):
    """
    Implements the 'Always' (G) temporal operator.
    
    This node performs a generalized conjunction across the time axis. 
    It uses the Gödel t-norm (minimum) to find the lower bound of truth 
    across all moments in the sequence.
    """
    
    def __init__(self, child: Node, window_size: int = None):
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the minimum truth interval across the temporal dimension.
        """
        a = self.child.forward(values)
        return jnp.min(a, axis=1)

class EventuallyNode(Node):
    """
    Implements the 'Eventually' (F) temporal operator.
    
    This node performs a generalized disjunction across the time axis. 
    It uses the maximum t-conorm to determine if the formula becomes 
    true at any point in the sequence.
    """

    def __init__(self, child: Node, window_size: int = None):
        self.child = child
        self.window_size = window_size

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the maximum truth interval across the temporal dimension.
        """
        a = self.child.forward(values)
        return jnp.max(a, axis=1)