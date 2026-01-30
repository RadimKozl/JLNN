#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.reasoning.inspector import trace_reasoning, get_rule_report
from jlnn.symbolic.compiler import LNNFormula

def test_trace_reasoning_structure(rngs):
    """
    Validates the recursive introspection of the logical computational graph.
    
    This test ensures that 'trace_reasoning':
    1. Correcty traverses the tree and identifies all participating nodes.
    2. Captures the specific naming convention (e.g., Predicate names).
    3. Provides a flat trace useful for Explainable AI (XAI) dashboards.
    """
    model = LNNFormula("A | B", rngs)
    inputs = {
        "A": jnp.array([[0.9, 1.0]]),
        "B": jnp.array([[0.1, 0.2]])
    }
    trace = trace_reasoning(model.root, inputs)
    
    # Verify the trace contains at least the root (OR) and two leaves (A, B)
    assert len(trace) >= 3
    node_names = [entry["name"] for entry in trace]
    # Check for Predicate name format
    assert any("Predicate(A)" in n for n in node_names)
    # Check for Gate presence
    assert any("Gate" in n for n in node_names)

def test_rule_report_classification(rngs):
    """
    Tests the semantic interpretation logic of the reasoning report.
    
    This test verifies the qualitative classification of truth intervals:
    1. High-confidence intervals are labeled as 'TRUE'.
    2. Low-confidence intervals are correctly identified, even with neural bias.
    3. The report handles the 'Neural False' case where weights shift [0, 0.05] to a neutral zone.
    """
    model = LNNFormula("A", rngs)
    
    # Scenario: Strong TRUE input
    report_true = get_rule_report(model, {"A": jnp.array([[0.95, 1.0]])})
    assert "TRUE" in report_true
    
    # Scenario: Strong FALSE input (testing robustness against neural shifting/squashing)
    report_false = get_rule_report(model, {"A": jnp.array([[0.0, 0.05]])})
    # We assert that the report contains 'FALSE' regardless of whether 
    # it is a pure logical FALSE or a NEURAL classification.
    assert "FALSE" in report_false