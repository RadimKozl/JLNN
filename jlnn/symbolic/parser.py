#!/usr/bin/env python3

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
    def __init__(self):
        self.parser = Lark(LNN_GRAMMAR, parser='lalr')

    def parse(self, formula: str):
        try:
            return self.parser.parse(formula)
        except Exception as e:
            raise ValueError(
                f"Chyba při parsování formule '{formula}': {str(e)}"
            )

if __name__ == "__main__":
    parser = FormulaParser()
    print(parser.parse("A & B | C").pretty())