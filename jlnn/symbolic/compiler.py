#!/usr/bin/env python3

"""
JLNN Symbolic Compiler - WORKING VERSION

The issue: Lark is looking for methods named "and", "or", "not" (from grammar rules)
but Python doesn't allow these as method names because they're keywords.
Solution: Use setattr() to dynamically create the attributes after class definition.
"""

from __future__ import annotations
from typing import Any, Dict, List, Union
from lark import Transformer, Tree, Token
from flax import nnx
import jax.numpy as jnp
from jlnn.nn import gates, predicates
from jlnn.symbolic.parser import FormulaParser
    
class Node(nnx.Module):
    """
    Abstract base class for all nodes in the JLNN computational graph.
    """
    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError

class PredicateNode(Node):
    """
    Represents a leaf node (variable) mapping to a LearnedPredicate.
    """
    def __init__(self, name: str, rngs: nnx.Rngs):
        self.name = name
        # Every variable gets its own trainable grounding
        self.predicate = predicates.LearnedPredicate(in_features=1, rngs=rngs)

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Input 'values' dictionary provides raw numeric data for this variable
        val = values[self.name]
        return self.predicate(val)

class NAryGateNode(Node):
    """
    Represents a gate with N inputs (AND, OR, XOR).
    """
    def __init__(self, gate: nnx.Module, children: List[Node]):
        self.gate = gate
        # Use nnx.List to store children that contain JAX arrays
        # This is required in Flax NNX 0.12+ for proper pytree handling
        self.children = nnx.List(children)

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Recursively evaluate all children and stack their results
        child_outputs = [child.forward(values) for child in self.children]
        # Stack on the second to last axis to match (batch, inputs, interval)
        x = jnp.stack(child_outputs, axis=-2)
        return self.gate(x)

class BinaryGateNode(Node):
    """
    Represents a gate with exactly 2 inputs (Implication).
    """
    def __init__(self, gate: nnx.Module, left: Node, right: Node):
        self.gate = gate
        self.left = left
        self.right = right

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        left_output = self.left.forward(values)
        right_output = self.right.forward(values)
        # The gate receives two separate arguments (antecedent, consequent)
        result = self.gate(left_output, right_output)
        # Remove any singleton dimensions that shouldn't be there
        if result.ndim == 3 and result.shape[1] == 1:
            result = jnp.squeeze(result, axis=1)
        return result

class UnaryGateNode(Node):
    """
    Represents a gate with 1 input (NOT).
    """
    def __init__(self, gate: nnx.Module, child: Node):
        self.gate = gate
        self.child = child

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return self.gate(self.child.forward(values))

class JLNNCompiler(Transformer):
    """
    Lark Transformer that converts a CST (Concrete Syntax Tree) into an NNX model.
    
    CRITICAL FIX: The grammar uses "and", "or", "not" as rule names, but these
    are Python keywords. We define methods with safe names (and_, or_, not_)
    and then use setattr() to create aliases with the exact names Lark expects.
    """
    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        self.rngs = rngs
        self.predicates: Dict[str, predicates.LearnedPredicate] = {}

    def variable(self, tokens):
        """Transform a variable token into a PredicateNode."""
        name = str(tokens[0])
        if name not in self.predicates:
            node = PredicateNode(name, self.rngs)
            self.predicates[name] = node
        return self.predicates[name]

    def and_(self, children):
        """Transform an AND operation."""
        gate = gates.WeightedAnd(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)
    
    def or_(self, children):
        """Transform an OR operation."""
        gate = gates.WeightedOr(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)
    
    def not_(self, children):
        """Transform a NOT operation."""
        gate = gates.WeightedNot(rngs=self.rngs)
        return UnaryGateNode(gate, children[0])

    def implication(self, children):
        """Transform an implication operation (A -> B)."""
        gate = gates.WeightedImplication(rngs=self.rngs)
        return BinaryGateNode(gate, children[0], children[1])

    def weighted_expr(self, children):
        """
        Processes the root node of the grammar.
        Returns the last child (the transformed expression).
        """
        return children[-1]

# CRITICAL: Use setattr to create aliases for Python keywords
# The grammar rules are: "-> and", "-> or", "-> not"
# Lark looks for methods with these exact names, but we can't use them directly
# because they're Python reserved words. So we use setattr() to assign them.
setattr(JLNNCompiler, 'and', JLNNCompiler.and_)
setattr(JLNNCompiler, 'or', JLNNCompiler.or_)
setattr(JLNNCompiler, 'not', JLNNCompiler.not_)

class LNNFormula(nnx.Module):
    """
    The main LNN Formula wrapper. Compiles a string into a neural structure.
    
    Example:
        >>> from flax import nnx
        >>> rngs = nnx.Rngs(0)
        >>> model = LNNFormula("A & B -> C", rngs)
        >>> inputs = {
        ...     "A": jnp.array([[0.9]]),
        ...     "B": jnp.array([[0.8]]),
        ...     "C": jnp.array([[0.1]])
        ... }
        >>> output = model(inputs)  # Returns truth interval [L, U]
    """
    def __init__(self, formula: str, rngs: nnx.Rngs):
        # Parse the formula string into an AST
        parser = FormulaParser()
        tree = parser.parse(formula)
        
        # Transform the AST into a computational graph
        compiler = JLNNCompiler(rngs)
        self.root = compiler.transform(tree)
        
        # Store predicates for access/inspection
        # For Flax NNX 0.12+, dict with parameters must be wrapped in nnx.Dict
        self.predicates = nnx.Dict(compiler.predicates)

    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Forward pass through the compiled logic tree.
        
        Args:
            inputs: Dictionary mapping variable names to their numeric values.
                   Each value should be a JAX array of shape (batch_size, features).
        
        Returns:
            JAX array of shape (batch_size, 2) representing truth intervals [L, U]
            where L is the lower bound and U is the upper bound of truth.
        """
        return self.root.forward(inputs)