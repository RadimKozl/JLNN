#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.training.losses import contradiction_loss, rule_violation_loss

def test_contradiction_loss_penalization():
    """
    Verifies that contradiction_loss penalizes intervals where Lower > Upper.

    Logical Consistency Axiom: In LNNs, it must hold that L <= U.
    - For valid intervals (e.g., [0.2, 0.8]), loss should be 0.
    - For contradictory intervals (e.g., [0.9, 0.1]), loss should be positive
      to push the model back into a valid logical state.
    """
    # Valid interval: No contradiction
    valid_int = jnp.array([[0.2, 0.8]])
    assert contradiction_loss(valid_int) == 0.0

    # Contradictory interval: L(0.9) > U(0.1)
    invalid_int = jnp.array([[0.9, 0.1]])
    loss = contradiction_loss(invalid_int)
    assert loss > 0.0, "Contradiction loss must be positive when L > U"

def test_rule_violation_loss_logic():
    """
    Tests the penalty for violating the implication semantics (A -> B).

    Implication Violation: Occurs when the antecedent is TRUE (high Lower bound) 
    but the consequent is FALSE (low Upper bound).
    
    Formula: max(0, L(A) - U(B))
    - Case A=TRUE, B=FALSE: High violation.
    - Case A=FALSE, B=FALSE: No violation.
    """
    # Case 1: No violation (A is False, B is False)
    ant_f = jnp.array([[0.0, 0.2]])
    con_f = jnp.array([[0.0, 0.1]])
    assert rule_violation_loss(ant_f, con_f) == 0.0

    # Case 2: Violation (A is True [0.9], B is False [0.1])
    # Expected penalty: 0.9 - 0.1 = 0.8
    ant_t = jnp.array([[0.9, 1.0]])
    con_f = jnp.array([[0.0, 0.1]])
    loss = rule_violation_loss(ant_t, con_f)
    
    assert jnp.isclose(loss, 0.8), f"Expected 0.8 violation, got {loss}"