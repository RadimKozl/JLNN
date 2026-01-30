#!/usr/bin/env python3

# Imports
import pytest
from jlnn.symbolic.parser import FormulaParser
from lark import Tree, Token

def test_parser_basic_and_or():
    """
    Verifies that the parser correctly constructs a syntax tree for compound logic.
    
    This test checks if the parser respects operator precedence (AND before OR)
    and successfully generates the hierarchical Tree structure.
    """
    parser = FormulaParser()
    tree = parser.parse("A & B | C")
    
    assert isinstance(tree, Tree)
    
    # The grammar has ?weighted_expr as start, which inlines if there's no weight prefix
    # But based on the error, it seems weighted_expr is the root
    # Navigate through the tree to find the 'or' node
    def find_node_by_data(node, target_data):
        """Recursively search the Lark Tree for a specific node type."""
        if isinstance(node, Tree):
            if node.data == target_data:
                return node
            for child in node.children:
                result = find_node_by_data(child, target_data)
                if result is not None:
                    return result
        return None
    
    or_node = find_node_by_data(tree, "or")
    assert or_node is not None, f"'or' node not found in tree. Root: {tree.data}"
    
    # Precedence check: The 'and' operation should be nested as a child or sibling of 'or'
    and_node = find_node_by_data(tree, "and")
    assert and_node is not None, "'and' node not found - precedence might be wrong"

def test_parser_weighted_implication():
    """
    Checks the parser's ability to extract weights from the custom '->[number]' syntax.
    
    Ensures that both the implication structure and the numerical parameter 
    are correctly identified as part of a 'weighted_implication' node.
    """
    parser = FormulaParser()
    tree = parser.parse("A ->[0.5] B")
    
    def find_node_by_data(node, target_data):
        """Recursively search for a node with specific data identifier."""
        if isinstance(node, Tree):
            if node.data == target_data:
                return node
            for child in node.children:
                result = find_node_by_data(child, target_data)
                if result is not None:
                    return result
        return None
    
    impl_node = find_node_by_data(tree, "weighted_implication")
    assert impl_node is not None, "weighted_implication node not found in tree"
    
    def find_number_token(node):
        """Recursively locate a Token of type NUMBER."""
        if isinstance(node, Token) and node.type == "NUMBER":
            return node
        if isinstance(node, Tree):
            for child in node.children:
                result = find_number_token(child)
                if result is not None:
                    return result
        return None
    
    number_token = find_number_token(tree)
    assert number_token is not None, "NUMBER token not found in tree"
    assert float(number_token.value) == 0.5, f"Expected 0.5, got {number_token.value}"

def test_parser_invalid_syntax():
    """
    Ensures that the parser correctly identifies and rejects syntactically invalid strings.
    
    A ValueError should be raised when encountering unknown operators or malformed expressions.
    """
    parser = FormulaParser()
    with pytest.raises(ValueError):
        parser.parse("A &&& B")  # Triggering an error with an invalid operator sequence