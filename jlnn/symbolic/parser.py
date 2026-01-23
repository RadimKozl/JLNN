#!/usr/bin/env python3

# Imports
from lark import Lark
import jax.numpy as jnp

# Grammar definition supporting standard, extended, and future temporal operators.
# The grammar is designed with operator precedence (binding) in mind.
LNN_GRAMMAR = r"""
    ?start: expr

    ?expr: term
         | expr "->" term    -> implication
         | expr "<->" term   -> equivalence

    ?term: factor
         | term "|" factor   -> or
         | term "^" factor   -> xor
         | term "!|" factor  -> nor

    ?factor: unary
           | factor "&" unary  -> and
           | factor "!&" unary -> nand

    ?unary: "~" unary        -> not
          | "G" unary        -> always
          | "F" unary        -> eventually
          | "X" unary        -> next
          | atom

    ?atom: NAME              -> variable
         | "(" expr ")"

    %import common.CNAME -> NAME
    %import common.WS
    %ignore WS
"""

class FormulaParser:
    """
    Advanced logical formula parser built on the Lark library.

    This module serves as a front-end for JLNN (Just-in-time Logical Neural Network). 
    Its task is to transform the textual notation of rules into 
    a structured tree (Concrete Syntax Tree), which can then be compiled into 
    a JAX/NNX computational graph.

    Main advantages:
    - Support for non-Python syntax: Direct use of operators like '->' 
    (impliction) or '!&' (NAND) without the need for string preprocessing.
    - Hierarchical reasoning: Properly defined operator precedence 
    (e.g. conjunction has a stronger link than disjunction).
    - Temporal logic preparation: Contains reserved tokens for 
    linear temporal logic operators (G, F, X).

    Grammarly supports:
    - Binary: AND (&), OR (|), XOR (^), NAND (!&), NOR (!|), Implication (->).
    - Unary: NOT (~), Always (G), Eventually (F), Next (X).
    """

    def __init__(self):
        """
        Initializes a parser instance with the defined grammar LNN_GRAMMAR.
        
        It uses the LALR(1) algorithm, which is optimal for deterministic logical grammars 
        and provides high parsing speed during model initialization.
        """
        self.parser = Lark(LNN_GRAMMAR, parser='lalr')

    def parse(self, formula: str):
        """
        It parses the input string and builds a syntax tree.

        Args:
            formula (str): Text representation of a logical expression (e.g. "A & B -> ~C").

        Returns:
            lark.Tree: Syntactic tree representing the structure of a formula.

        Raises:
            ValueError: If the formula does not conform to the defined grammar 
            or contains illegal characters.
        """
        try:
            return self.parser.parse(formula)
        except Exception as e:
            # Catching errors from Lark and transforming them into a more understandable exception
            raise ValueError(f"Error parsing formula '{formula}': {e}")