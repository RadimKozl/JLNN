import jax.numpy as jnp
from jlnn.reasoning.inspector import trace_reasoning, get_rule_report
from jlnn.symbolic.compiler import LNNFormula

def test_trace_reasoning_structure(rngs):
    model = LNNFormula("A | B", rngs)
    inputs = {
        "A": jnp.array([[0.9, 1.0]]),
        "B": jnp.array([[0.1, 0.2]])
    }
    trace = trace_reasoning(model.root, inputs)
    
    assert len(trace) >= 3
    node_names = [entry["name"] for entry in trace]
    # Check for Predicate name format
    assert any("Predicate(A)" in n for n in node_names)
    # Check for Gate presence
    assert any("Gate" in n for n in node_names)

def test_rule_report_classification(rngs):
    model = LNNFormula("A", rngs)
    
    # Test High Confidence TRUE
    report_true = get_rule_report(model, {"A": jnp.array([[0.95, 1.0]])})
    assert "TRUE" in report_true
    
    # Test FALSE (handling neural shifting)
    report_false = get_rule_report(model, {"A": jnp.array([[0.0, 0.05]])})
    # We assert that the report contains 'FALSE' regardless of whether 
    # it is a pure logical FALSE or a NEURAL classification.
    assert "FALSE" in report_false