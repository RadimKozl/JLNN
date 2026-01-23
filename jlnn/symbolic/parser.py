#!/usr/bin/env python3

# Imports
from lark import Lark
import jax.numpy as jnp

# Grammar definition supporting fact weights, weighted rules, and future tense operators.
# Operator precedence is enforced by hierarchy (expr -> term -> factor -> unary -> atom).
LNN_GRAMMAR = r"""
    ?start: weighted_expr

    # Allows you to define the a priori truth of a fact (prefix): 0.9::A
    ?weighted_expr: [NUMBER "::"] expr 

    # Binary operators with lowest precedence
    ?expr: term
         | expr "->" term                -> implication
         | expr "->" "[" NUMBER "]" term  -> weighted_implication
         | expr "<->" term               -> equivalence

    # Disjunctive operators
    ?term: factor
         | term "|" factor               -> or
         | term "^" factor               -> xor
         | term "!|" factor              -> nor

    # Conjunctive operators
    ?factor: unary
           | factor "&" unary            -> and
           | factor "!&" unary           -> nand

    # Unary operators (including future temporal logic)
    ?unary: "~" unary                    -> not
          | "G" unary                    -> always
          | "F" unary                    -> eventually
          | "X" unary                    -> next
          | atom

    # Basic building blocks
    ?atom: NAME                          -> variable
         | "(" expr ")"

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

class FormulaParser:
    """
    An advanced parser for weighted neuro-symbolic logic in the JAX ecosystem.

    This class transforms textual logical formulas into an abstract syntax tree (AST), 
    which in JLNN serves as a blueprint for building a computational graph. 
    Unlike standard parsers, it allows you to insert numerical parameters into 
    the text to initialize the network weights.

    ### Supported semantics:
    1. **Fact weights (`0.9::A`)**: Allows you to define the initial truth value of the predicate. 
                                In `predicates.py`, this value is used to 
                                set the initial offsets of the sigmoid.
    2. **Weighted rules (`->[0.85]`)**: Defines the strength of the logical connection between 
                                the antecedent and consequent. This value maps directly to the `weights` parameter 
                                in `WeightedImplication`.
    3. **Complete gate set**: Support for AND, OR, XOR, NAND, NOR.
    4. **Temporal preparation**: The grammar contains reserved operators G (Always), 
                                F (Eventually), and X (Next) for linear temporal logic (LTL).

    The parser uses the LALR(1) algorithm for deterministic and lightning-fast 
    parsing during model initialization.
    """

    def __init__(self):
        """
        Initializes an instance of the Lark parser with the defined LNN grammar.
        """
        self.parser = Lark(LNN_GRAMMAR, parser='lalr')

    def parse(self, formula: str):
        """
        Converts a text formula into a structured syntax tree.

        This tree (Lark Tree) is then processed by a compiler, 
        which creates a corresponding JAX/NNX module (gateway) for each node.

        Args:
            formula (str): A logical expression, e.g. "0.7::(A & B) ->[0.9] C".

        Returns:
            lark.Tree: Syntax tree with extracted identifiers and numbers.

        Raises:
            ValueError: If the formula violates grammar rules 
                    or contains invalid numeric literals.
        """
        try:
            return self.parser.parse(formula)
        except Exception as e:
            # Catching syntax errors and interpreting them in a clear way
            raise ValueError(
                f"Error parsing weighted formula '{formula}'. "
                f"Check the syntax and numerical weights. Details: {e}"
            )