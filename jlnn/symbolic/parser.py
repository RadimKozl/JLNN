#!/usr/bin/env python3

"""
This module provides a parser for Logical Neural Networks (LNN) formulas.
It utilizes the Lark library to transform logical expressions into a parse tree,
supporting standard logic gates, temporal operators, and weighted expressions.
"""

# Imports
from lark import Lark

LNN_GRAMMAR = r"""
    ?start: weighted_expr

    # Otazník zajistí, že pokud NUMBER:: chybí, uzel se zploští
    ?weighted_expr: [NUMBER "::"] expr 

    ?expr: term
         | expr "->" term                -> implication
         | expr "->" "[" NUMBER "]" term  -> weighted_implication
         | expr "<->" term               -> equivalence

    ?term: factor
         | term "|" factor               -> or
         | term "^" factor               -> xor
         | term "!|" factor              -> nor

    ?factor: unary
           | factor "&" unary            -> and
           | factor "!&" unary           -> nand

    ?unary: "~" unary                    -> not
          | "G" unary                    -> always
          | "F" unary                    -> eventually
          | "X" unary                    -> next
          | atom

    ?atom: NAME                          -> variable
         | "(" expr ")"

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

class FormulaParser:
    """
    A wrapper class for the Lark LALR parser configured with LNN grammar.
    
    This parser handles operator precedence and associativity for logical 
    operators, including support for weighted implications and temporal 
    logic operators like Always (G), Eventually (F), and Next (X).
    """
    def __init__(self):
        """
        Initializes the FormulaParser by compiling the LNN grammar 
        using the LALR parsing algorithm.
        """
        self.parser = Lark(LNN_GRAMMAR, parser='lalr')

    def parse(self, formula: str):
        """
        Parses a string representation of a logical formula into a Lark Tree.

        Args:
            formula (str): The logical expression to be parsed (e.g., "A & B -> C").

        Returns:
            lark.Tree: A parse tree representing the hierarchical structure of the formula.

        Raises:
            ValueError: If the input formula does not conform to the LNN grammar rules.
        """
        try:
            return self.parser.parse(formula)
        except Exception as e:
            raise ValueError(
                f"Chyba při parsování formule '{formula}': {str(e)}"
            )

if __name__ == "__main__":
    parser = FormulaParser()
    print(parser.parse("A & B | C").pretty())