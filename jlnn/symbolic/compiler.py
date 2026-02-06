#!/usr/bin/env python3

"""
JLNN Symbolic Compiler

This module provides the infrastructure to compile symbolic logical formulas 
into neural computational graphs using Flax NNX and JAX. It leverages 
the Lark parser to transform strings into a tree of NNX Modules.
"""

# Imports
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
    
    Each node represents a logical operation or a predicate and must 
    implement a forward pass that operates on JAX arrays.
    """
    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Performs the forward evaluation of the node.

        Args:
            values: A mapping of variable names to JAX arrays containing 
                   the raw input features.

        Returns:
            A JAX array representing the truth value (typically a truth interval).
        """
        raise NotImplementedError

class PredicateNode(Node):
    """
    Represents a leaf node (variable) mapping to a LearnedPredicate.
    
    This node acts as the grounding layer where raw numeric data is 
    transformed into a fuzzy truth value.
    """
    def __init__(self, name: str, rngs: nnx.Rngs):
        """
        Initializes the predicate node.

        Args:
            name: The identifier of the variable in the formula.
            rngs: Flax NNX random number generator stream for parameter initialization.
        """
        self.name = name
        # Every variable gets its own trainable grounding (LearnedPredicate)
        self.predicate = predicates.LearnedPredicate(in_features=1, rngs=rngs)

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Retrieves input data by name and passes it through the predicate."""
        val = values[self.name]
        return self.predicate(val)

class NAryGateNode(Node):
    """
    Represents a logic gate with N inputs, such as weighted AND, OR or XOR.
    """
    def __init__(self, gate: nnx.Module, children: List[Node]):
        """
        Args:
            gate: The neural logic gate module (e.g., WeightedAnd).
            children: A list of child Nodes whose outputs are inputs to this gate.
        """
        self.gate = gate
        # nnx.List is required for Flax NNX to correctly track child modules 
        # and their parameters during JAX transformations.
        self.children = nnx.List(children)

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Evaluates all children, stacks results, and applies the gate logic."""
        child_outputs = [child.forward(values) for child in self.children]
        # Stack results on the second-to-last axis to form a tensor of 
        # shape (batch, num_inputs, truth_interval_dims).
        x = jnp.stack(child_outputs, axis=-2)
        return self.gate(x)

class BinaryGateNode(Node):
    """
    Represents a gate with exactly 2 inputs, specifically designed for 
    asymmetric operations like Implication (A -> B).
    """
    def __init__(self, gate: nnx.Module, left: Node, right: Node):
        """
        Args:
            gate: The binary logic gate module.
            left: The antecedent (left-hand side) node.
            right: The consequent (right-hand side) node.
        """
        self.gate = gate
        self.left = left
        self.right = right

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Evaluates antecedent and consequent separately before applying the gate."""
        left_output = self.left.forward(values)
        right_output = self.right.forward(values)
        
        # The gate receives two separate arguments (antecedent, consequent)
        result = self.gate(left_output, right_output)
        
        # Squeeze singleton dimensions to maintain consistent output shape (batch, 2)
        if result.ndim == 3 and result.shape[1] == 1:
            result = jnp.squeeze(result, axis=1)
        return result

class UnaryGateNode(Node):
    """
    Represents a gate with a single input, such as NOT.
    """
    def __init__(self, gate: nnx.Module, child: Node):
        """
        Args:
            gate: The unary logic gate module.
            child: The child node to be negated.
        """
        self.gate = gate
        self.child = child

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Passes the child's output through the unary gate."""
        return self.gate(self.child.forward(values))

class JLNNCompiler(Transformer):
    """
    Lark Transformer that converts a CST (Concrete Syntax Tree) into an NNX model tree.
    
    Note:
        The grammar rules use ``and``, ``or``, and ``not``. Since these are reserved 
        keywords in Python, we define them as ``and_``, ``or_``, ``not_`` and alias them 
        dynamically after class definition.
    """
    def __init__(self, rngs: nnx.Rngs):
        """
        Args:
            rngs: Random streams for initializing gate and predicate parameters.
        """
        super().__init__()
        self.rngs = rngs
        self.predicates: Dict[str, predicates.LearnedPredicate] = {}

    def variable(self, tokens: List[Token]) -> PredicateNode:
        """Transforms a variable token into a PredicateNode, ensuring weight sharing if name repeats."""
        name = str(tokens[0])
        if name not in self.predicates:
            node = PredicateNode(name, self.rngs)
            self.predicates[name] = node
        return self.predicates[name]

    def and_(self, children: List[Node]) -> NAryGateNode:
        """Constructs a WeightedAnd gate node."""
        gate = gates.WeightedAnd(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)
    
    def or_(self, children: List[Node]) -> NAryGateNode:
        """Constructs a WeightedOr gate node."""
        gate = gates.WeightedOr(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)
    
    def not_(self, children: List[Node]) -> UnaryGateNode:
        """Constructs a WeightedNot gate node."""
        gate = gates.WeightedNot(rngs=self.rngs)
        return UnaryGateNode(gate, children[0])

    def implication(self, children: List[Node]) -> BinaryGateNode:
        """Constructs a WeightedImplication gate node (A -> B)."""
        gate = gates.WeightedImplication(rngs=self.rngs)
        return BinaryGateNode(gate, children[0], children[1])

    def weighted_expr(self, children: List[Any]) -> Node:
        """Root rule processor; returns the final compiled node structure."""
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
    The main LNN Formula wrapper. 
    Compiles a logical string formula into a differentiable NNX module.
    
    Attributes:
        root: The root node of the neural computational graph.
        predicates: An nnx.Dict containing all trainable predicate groundings.
    
    Example:
        >>> rngs = nnx.Rngs(0)
        >>> model = LNNFormula("A & B -> C", rngs)
        >>> inputs = {"A": jnp.array([[0.9]]), "B": jnp.array([[0.8]]), "C": jnp.array([[0.1]])}
        >>> output = model(inputs)  # Returns truth interval [Lower, Upper]
    """
    def __init__(self, formula: str, rngs: nnx.Rngs):
        """
        Parses and compiles the formula.

        Args:
            formula: The logical formula string (e.g., 'A & B').
            rngs: Flax NNX random number generator stream.
        """
        # Parse the formula string into a Concrete Syntax Tree (CST)
        parser = FormulaParser()
        tree = parser.parse(formula)
        
        # Transform the CST into a tree of NNX Modules (Nodes)
        compiler = JLNNCompiler(rngs)
        self.root = compiler.transform(tree)
        
        # Wrap the predicates dictionary in nnx.Dict for proper NNX parameter tracking
        self.predicates = nnx.Dict(compiler.predicates)

    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Executes the forward pass of the compiled logic tree.
        
        Args:
            inputs: Dictionary mapping variable names to JAX arrays of shape (batch_size, features).
        
        Returns:
            A JAX array of shape (batch_size, 2) representing truth intervals [L, U].
        """
        return self.root.forward(inputs)