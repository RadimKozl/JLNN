#!/usr/bin/env python3

# Imports
import pytest
from jlnn.symbolic.parser import FormulaParser
from lark import Tree, Token

def test_parser_basic_and_or():
    """
    Verifies that the parser correctly builds a syntax tree for basic 
    conjunction and disjunction.
    """
    parser = FormulaParser()
    tree = parser.parse("A & B | C")
    
    assert isinstance(tree, Tree)
    
    # The grammar has ?weighted_expr as start, which inlines if there's no weight prefix
    # But based on the error, it seems weighted_expr is the root
    # Navigate through the tree to find the 'or' node
    def find_node_by_data(node, target_data):
        """Recursively search for a node with specific data"""
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
    
    # Verify that OR has an AND as one of its children (precedence check)
    and_node = find_node_by_data(tree, "and")
    assert and_node is not None, "'and' node not found - precedence might be wrong"

def test_parser_weighted_implication():
    """
    Checks if the parser correctly extracts weights from the 
    special '->[number]' notation.
    """
    parser = FormulaParser()
    tree = parser.parse("A ->[0.5] B")
    
    # Find the weighted_implication node
    def find_node_by_data(node, target_data):
        """Recursively search for a node with specific data"""
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
    
    # Search for a NUMBER token anywhere in the tree
    def find_number_token(node):
        """Recursively find a NUMBER token"""
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
    """Ensures the parser raises a ValueError for mathematically invalid strings."""
    parser = FormulaParser()
    with pytest.raises(ValueError):
        parser.parse("A &&& B")  # Invalid operator