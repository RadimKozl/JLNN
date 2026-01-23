#!/usr/bin/env python3

import ast

# Symbol mapping to JLNN gates
SYMBOL_MAP = {
    '&': 'WeightedAnd',          # Conjunction
    '|': 'WeightedOr',           # Disjunction
    '~': 'WeightedNot',          # Negation
    '^': 'WeightedXor',          # Exclusive disjunction
    '!&': 'WeightedNand',        # Negated conjunction
    '!|': 'WeightedNor',         # Negated disjunction
    '->': 'WeightedImplication'  # Implication
}

class FormulaParser:
    """
    Parses textual logical expressions and converts them to an AST structure for JLNN.

    This parser serves as a bridge between the symbolic notation of rules 
    and the computational graph of the neural network. 
    It supports standard logical operators as well 
    as specific extended operators (NAND, NOR, Implication).

    It uses the Python AST for safe parsing, performing pre-transformation of symbols that 
    are not native to Python syntax.
    """
    
    def __init__(self):
        """Initializes a parser instance."""
        pass

    def _preprocess_formula(self, formula: str) -> str:
        """
        Transforms non-Python operators into temporary valid tokens.
        
        E.g. 'A -> B' changes to 'A >> B' (bit shift) to make it 
        ast.parse was able to process it as a binary operation.
        """
        f = formula.replace(" ", "")
        # Transformation to maintain compatibility with the ast module
        f = f.replace("->", ">>")  # Implication simulated as a bit shift right
        f = f.replace("!&", "@")   # NAND simulated as matrix multiplication
        f = f.replace("!|", "==")  # NOR simulated as comparison
        return f

    def parse_to_ast(self, formula: str) -> ast.Expression:
        """
        Converts a text string to a Python AST (Abstract Syntax Tree).

        Args:
            formula (str): A text string with a logical formula (e.g. "(A & B) | ~C").

        Returns:
            ast.Expression: The root node of the abstract syntax tree.

        Raises:
            ValueError: If the formula contains syntax errors.
        """
        clean_formula = self._preprocess_formula(formula)
        try:
            # 'eval' mode is ideal for parsing single-line expressions
            tree = ast.parse(clean_formula, mode='eval')
            return tree
        except SyntaxError as e:
            raise ValueError(f"Invalid logical formula: {formula}. Error: {e}")

    def extract_variables(self, tree: ast.AST) -> list[str]:
        """
        It traverses the AST and extracts a list of all unique variable names.

        These variables will subsequently be replaced in JLNN 
        by instances of the LearnedPredicate class.

        Args:
            tree (ast.AST): Tree obtained by the parse_to_ast function.

        Returns:
            list[str]: Sorted list of unique identifiers (e.g. ['A', 'B']).
        """
        variables = set()
        for node in ast.walk(tree):
            # In AST mode 'eval', variables are represented by ast.Name nodes
            if isinstance(node, ast.Name):
                variables.add(node.id)
        return sorted(list(variables))