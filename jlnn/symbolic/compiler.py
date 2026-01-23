#!/usr/bin/env python3

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
    Abstract base class for all nodes of the JLNN computational graph.
    
    This class defines a uniform interface for recursively evaluating compiler-generated logic trees. 
    Each node in the graph represents either:
    1. **List (Atom)**: Instance of `PredicateNode` (input data transformed to truth).
    2. **Operation (Gate)**: Instance of `BinaryGateNode` or `NAryGateNode` (logical connectors).

    Thanks to inheritance from `nnx.Module`, the entire resulting tree is perceived by the Flax library as 
    a single hierarchical model. This allows:
    - **Automatic parameter tracking**: All weights and biases inside arbitrarily 
                                    deeply embedded nodes are detectable to the optimizer.
    - **Application of constraints**: The `apply_constraints` function can recursively traverse 
                                    the entire structure and fix gate weights.
    - **JIT Compilation**: The entire recursive traversal can be compiled 
                        using `jax.jit` for maximum performance on GPU/TPU.
    """

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Recursively calculates the truth value of a given subtree (inference).

        The method implements a depth-first search. Each gate node first calls `forward` 
        on its children and then applies its logical function (t-norm/t-conorm) to the results.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary (map) of input data, 
                                    where keys are variable names from a logical formula (e.g. 'A', 'B') 
                                    and values ​​are JAX tensors (typically of the form [batch_size, features]).

        Returns:
            jnp.ndarray: Tensor of truth values ​​in the interval [0, 1] (LNN fuzzy logic). 
                    The output usually has the form [batch_size], representing the truth value 
                    of the given (sub)formula for each instance in the batch.

        Raises:
            NotImplementedError: If the method is called directly on the abstract class Node. 
                        KeyError: If the input dictionary 'values' does not contain 
                        a key required by the leaf node (predicate).
        """
        raise NotImplementedError

class PredicateNode(Node):
    """
    The terminal node (leaf) of a computational graph representing a learned logical predicate.
    
    This node serves as the data entry gateway into the logical network. 
    Its main task is to take a raw numerical value (e.g., sensory data or probability) 
    and transform it into a logical interval [L, U] (Lower and Upper bound) 
    using the `LearnedPredicate` layer.

    Features:
    - **Learning semantics**: Thanks to the nested `LearnedPredicate`, 
        the node learns how to interpret the input data to best fit the logical rules 
        defined in the rest of the graph.
    - **Knowledge Initialization**: Supports setting an initial bias, 
        which allows a priori expert knowledge to be inserted into the network 
        (e.g. via `0.9::A` notation in the parser).
    - **Variable Isolation**: Each unique variable in a formula has its own `PredicateNode`, 
        allowing independent learning of the interpretation for each symbol.
    """

    def __init__(self, name: str, rngs: nnx.Rngs, initial_bias: float = 0.0):
        """
        Initializes the predicate node and its teachable transformation function.

        Args:
            name (str): Variable identifier corresponding to the name in the logical formula (e.g. "A").
            rngs (nnx.Rngs): Random number generator for initializing predicate parameters.
            initial_bias (float): Initial bias value. Used to project fact weights from the parser 
                        (e.g. from '0.9::A') into the initial state of the predicate.
        """
        self.name = name
        # Initialize the learning layer itself from predicates.py
        # in_features=1, because the predicate typically maps one scalar value to a logical interval
        self.predicate = predicates.LearnedPredicate(
            in_features=1,
            rngs=rngs,
            # Note: initial_bias is used here to initialize the bias_l and bias_u parameters
        )

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        It extracts the appropriate value from the inputs and performs logical grounding.

        Takes the tensor corresponding to `self.name` from the `values` dictionary 
        and runs it through the sigmoidal predicate transform.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary containing mappings 
                                    of variable names to input JAX fields.

        Returns:
            jnp.ndarray: Output tensor representing the logical interval [L, U]. 
                    The output has the form [batch_size, 2], 
                    where index 0 is the lower bound (L) and index 1 is the upper bound (U).
        """
        # Get input for this specific variable
        x = values[self.name]
        
        # LearnedPredicate expects input in the form (batch, features),
        # so we add a dimension using [..., None]
        return self.predicate(x[..., None])

class UnaryGateNode(Node):
    """
    A computational node representing a unary logical operation in a JLNN graph.

    This node acts as a wrapper for unary gates that accept only one input logical interval. 
    Within the Just-in-time Logical Neural Network architecture, 
    it is necessary for implementing operations that modify the truth value of a subordinate expression.

    Main use cases:
    1. **Logical Negation**: Typically uses the `WeightedNot` gate to invert 
        the truth interval [L, U] to [1-U, 1-L].
    2. **Temporal operators**: Serves as a basis for future implementation of the "Always" (G), 
        "Sometimes" (F), or "Next" (X) operators, which are already reserved in the defined grammar.

    Thanks to encapsulation in `nnx.Module`, the parameters of the nested gate (e.g. the weights in `WeightedNot`) 
    are fully trainable and subject to global constraints using `apply_constraints`.
    """

    def __init__(self, gate: nnx.Module, child: Node):
        """
        Initializes a unary node with a specific gate and its child.

        Args:
            gate (nnx.Module): An instance of a unary gate (e.g. from the gates.py module) 
                    that performs the actual mathematical calculation of the logical function.
            child (Node): A child node (descendant) in the logical 
                    tree whose output will be processed by this gate.
        """
        self.gate = gate
        self.child = child
        
    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Recursively evaluates the child and applies the unary operation.

        The method first delves deeper into the graph by calling `forward` 
        on the `self.child` object and then passes the obtained 
        truth interval (tensor) to the assigned gate.

        Args:
            values ​(Dict[str, jnp.ndarray]): Dictionary of input data mapping variable names to JAX tensors.

        Returns:
            jnp.ndarray: The resulting transformed logical interval of the form [batch_size, 2].
        """
        # Recursive descent to a descendant
        a = self.child.forward(values)
        
        # Application of unary logical function (e.g. negation)
        return self.gate(a)

class BinaryGateNode(Node):
    """
    A computational node representing a binary logical operation in a JLNN graph.

    This node is used to connect two subtrees using a logic gate that requires exactly two inputs. 
    In the Just-in-time Logical Neural Network architecture, 
    this node is essential for operations where the semantics 
    are defined by the relationship between the left (antecedent) and right (consequent) sides.

    Typical use cases:
    1. **Implication (->)**: Where the left side conditions the right side.
    2. **Equivalence (<->)**: Expressing two-way logical agreement.
    3. **Exclusive disjunction (XOR / ^)**: Where the result depends on the difference of the inputs.

    Thanks to its fixed structure (left/right), this node allows for precise mapping of weights to specific inputs, 
    which is especially critical for weighted implications defined in the parser as `->[weight]`.
    """

    def __init__(self, gate: nnx.Module, left: Node, right: Node):
        """
        Initializes a binary node with a specific gate and two children.

        Args:
            gate (nnx.Module): Binary gate instance from the gates.py module 
                (e.g. WeightedImplication, WeightedXor).
            left (Node): The left child (subtree) of a logical expression.
            right (Node): The right child (subtree) of a logical expression.
        """
        self.gate = gate
        self.left = left
        self.right = right

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        It recursively evaluates both subtrees and applies a binary logical function.

        The method runs the calculation for both the left and right branches in parallel 
        (within the JAX graph) and passes the resulting truth intervals to the gate for processing.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary of input data mapping variable names to JAX tensors.

        Returns:
            jnp.ndarray: The resulting logical interval [L, U] of the form [batch_size, 2] 
                resulting from the interaction of both branches.
        """
        # Recursive evaluation of left and right branches
        a = self.left.forward(values)
        b = self.right.forward(values)
        
        # Calculation of a binary operation using a gate (e.g. Lukasiewicz implication)
        return self.gate(a, b)

class NAryGateNode(Node):
    """
    A computational node representing a logical operation with a variable number of inputs.

    This node is used to efficiently group multiple conjunctions (AND), 
    disjunctions (OR), or their negated variants (NAND, NOR) into a single operation. 
    In the JLNN architecture, this approach is preferred over binary chaining because 
    it reduces the depth of recursion and allows the gate to process all arguments 
    at once using matrix operations.

    Main advantages:
    - **Computational efficiency**: Aggregating inputs into a single tensor (`jnp.stack`) 
        allows for fast gate computation on GPU/TPU.
    - **Structural clarity**: Logical expressions of type 'A & B & C & D' 
        are represented by a single node with four children instead of 
        a cascade of three binary nodes.
    - **Weight Flexibility**: A gate inside this node (e.g. `WeightedAnd`) 
        automatically adapts the number of its learnable weights to the number of descendants.

    Thanks to inheritance from `nnx.Module`, the weights assigned to individual inputs 
    are automatically optimized during the training process.
    """

    def __init__(self, gate: nnx.Module, children: List[Node]):
        """
        Initializes an n-ary node with a bulk gate and a list of descendants.

        Args:
            gate (nnx.Module): A bulk gate instance from the gates.py module that requires 
                            a 'num_inputs' parameter (e.g. WeightedAnd, WeightedOr).
            children (List[Node]): List of all child nodes (arguments) 
                            whose truth values ​​enter the operation.
        """
        self.gate = gate
        self.children = children

    def forward(self, values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Recursively evaluates all descendants and performs a bulk logical operation.

        The method first collects the truth intervals [L, U] from all descendants, 
        folds them into a single data structure along the new axis, and passes the result to the gate.

        Args:
            values ​​(Dict[str, jnp.ndarray]): Dictionary of input data mapping variable names to JAX tensors.

        Returns:
            jnp.ndarray: The resulting logical interval [L, U] 
                representing the aggregate truth of the entire group of operations.
        """
        # Recursively get the results of all children (list of tensors of the form [batch, 2])
        inputs = [ch.forward(values) for ch in self.children]
        
        # Fold the inputs into a tensor of the form [batch, 2, num_children]
        # This transformation is key for vectorized computations in gates
        stacked = jnp.stack(inputs, axis=-1)
        
        # Calculation of n-ary logical function (e.g. weighted product AND)
        return self.gate(stacked)

class JLNNCompiler(Transformer):
    """
    A compiler transforming a syntax tree (CST) into a hierarchical NNX model.

    This class implements the Visitor design pattern, which recursively traverses 
    the tree generated by the parser. However, instead of simply traversing, 
    it performs "backsynthesizing":
    1. **Sheets** (variables) are converted to `PredicateNode` instances.
    2. **Branches** (operators) are converted to the appropriate `GateNode` nodes.
    3. Insert **Weights** extracted from the text (e.g. 0.9) into the gate parameters.

    The result of the transformation is an object of type `Node`, 
    which is the root of the entire computational graph and at the same time a standard `flax.nnx.Module`.
    """

    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the compiler and prepares registers for graph management.

        Args:
            rngs (nnx.Rngs): A collection of random number generators for deterministic initialization 
                of the weights of all gates in the graph.
        """
        self.rngs = rngs
        self.predicates: Dict[str, PredicateNode] = {}
        self.fact_weights: Dict[str, float] = {}

    def _get_or_create_predicate(self, name: str, initial_weight: float = 0.5) -> PredicateNode:
        """
        It ensures the uniqueness of predicates for variables with the same name.

        If the variable 'A' occurs multiple times in a formula, 
        the compiler ensures that all occurrences share the same `LearnedPredicate` instance. 
        This is crucial for logical consistency of the model.

        Args:
            name (str): Variable name.
            initial_weight (float): Probability in the range [0, 1] used to initialize the bias.

        Returns:
            PredicateNode: The node representing the given variable.
        """
        if name not in self.predicates:
            # Convert probability [0,1] to logit space for bias.
            # The logit transformation allows the sigmoid to start at a defined level.
            bias = float(jnp.log(initial_weight / (1.0 - initial_weight + 1e-6)))
            self.predicates[name] = PredicateNode(name, self.rngs, initial_bias=bias)
        return self.predicates[name]

    def variable(self, tokens: List[Token]) -> Node:
        """
        Processes the terminal node of a variable.

        Args:
            tokens (List[Token]): A list of tokens from Lark, where the first is the variable name.
        """
        name = str(tokens[0])
        weight = self.fact_weights.get(name, 0.5)
        return self._get_or_create_predicate(name, weight)

    def and_(self, children: List[Node]) -> Node:
        """
        It builds a conjunctive node (Weighted AND).
        It allows n-ary connection of multiple subtrees into a single gate.
        """
        gate = gates.WeightedAnd(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)

    def or_(self, children: List[Node]) -> Node:
        """
        Constructs a disjunctive node (Weighted OR).
        Aggregates any number of inputs into a single disjunctive gate.
        """
        gate = gates.WeightedOr(num_inputs=len(children), rngs=self.rngs)
        return NAryGateNode(gate, children)

    def not_(self, children: List[Node]) -> Node:
        """
        Constructs a negation node (Weighted NOT).
        It is taking over one offspring.
        """
        gate = gates.WeightedNot(rngs=self.rngs)
        return UnaryGateNode(gate, children[0])

    def implication(self, children: List[Node]) -> Node:
        """
        It constructs a standard binary implication (A -> B).
        """
        gate = gates.WeightedImplication(rngs=self.rngs)
        return BinaryGateNode(gate, children[0], children[1])

    def weighted_implication(self, children: List[Any]) -> Node:
        """
        It constructs a weighted implication with a specific rule strength.

        It uses a numeric value enclosed in square brackets (e.g. ->[0.85]) 
        to directly initialize the gate weights.

        Args:
            children: List [antecedent, weight, consequent].
        """
        antecedent, w_token, consequent = children
        weight = float(w_token)
        gate = gates.WeightedImplication(rngs=self.rngs)
        # Initialize gate weights: first weight corresponds to antecedent
        gate.weights = nnx.Param(jnp.array([weight, 1.0]))
        return BinaryGateNode(gate, antecedent, consequent)

    def weighted_expr(self, children: List[Any]) -> Node:
        """
        Processes a root expression with an optional fact weight (e.g. 0.9::A).

        If the expression contains a weight, this method captures it 
        and can be used in the future for global calibration of the formula's truthfulness.
        """
        if len(children) == 1:
            return children[0]
        
        # children[0] is the fact weight, children[1] is the resulting node of the expression
        weight, expr = float(children[0]), children[1]
        return expr

class JLNNModel(nnx.Module):
    """
    Complete Just-in-time Logical Neural Network (JLNN) model.

    This class represents a unifying wrapper that connects symbolic logic to 
    a differentiable computational graph. The user defines the model using natural logical notation, 
    which is automatically transformed into a hierarchical gate structure based on JAX.

    Main responsibilities:
    1. **Orchestration**: Controls the flow from the text formula through the syntax tree (Lark) 
            to the instantiation of computing nodes (Node).
    2. **State Management**: Maintains references to all learnable predicates and logic gates, 
            allowing Flax NNX to track parameters for optimization.
    3. **Unified Inference**: Provides a simple `__call__` interface for processing batch data 
            (tensor dictionaries) in a single forward pass.

    The model is fully compatible with `jax.jit`, `jax.grad` 
    and methods for applying logical constraints.
    """

    def __init__(self, formula: str, rngs: nnx.Rngs):
        """
        Initializes the JLNN model based on the specified logical formula.

        During initialization, the text is parsed once and the entire network is built. 
        Although this process is symbolic, the resulting graph is purely numerical.

        Args:
            formula (str): A logical formula defining the structure 
                    of the network (e.g. "0.8::A & B ->[0.9] C").
            rngs (nnx.Rngs): A collection of random number generators 
                    for initializing weights and biases throughout the model.
        """
        # 1. Step: Transforming the text into a syntax tree (CST)
        parser = FormulaParser()
        tree = parser.parse(formula)
        
        # 2. Step: Transform the tree into a hierarchy of JAX/NNX modules
        self.compiler = JLNNCompiler(rngs)
        self.root = self.compiler.transform(tree)
        
        # 3. Step: Register unique predicates for easy access to parameters
        self.predicates = self.compiler.predicates

    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Performs a forward pass (inference) through the compiled logic graph.

        It takes the raw data for each symbol and recursively 
        calculates the truth value of the entire formula. 
        Thanks to JAX, this process is vectorized over the entire batch of data.

        Args:
            inputs (Dict[str, jnp.ndarray]): A dictionary where the keys correspond to the names 
            of the variables in the formula and the values ​​are tensors of the corresponding features.

        Returns:
            jnp.ndarray: The resulting logical interval [L, U] for the entire formula. 
                    The output format is typically [batch_size, 2].
        """
        # Run recursive forward pass from root node
        return self.root.forward(inputs)