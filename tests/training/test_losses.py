#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.training.losses import contradiction_loss, rule_violation_loss, jlnn_learning_loss

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
    
    
def test_jlnn_learning_loss_components():
    """
    Verifies that jlnn_learning_loss correctly combines all three learning objectives:
    1. MSE (Accuracy)
    2. Contradiction penalty (Consistency)
    3. Uncertainty penalty (Decisiveness/Width)
    """
    # Base case: Prediction matches Target exactly, and is a sharp point [1, 1]
    # In this case, MSE=0, Contra=0, Uncertainty (1-1)=0 -> Total loss = 0
    target = jnp.array([[1.0, 1.0]])
    prediction_perfect = jnp.array([[1.0, 1.0]])
    assert jnp.isclose(jlnn_learning_loss(prediction_perfect, target), 0.0)

    # Case: Prediction is perfectly accurate (MSE=0) and valid, but uncertain [0, 1]
    # Target is [0.5, 0.5]. Prediction [0, 1] has MSE=0.25 (dist to 0.5), no contra,
    # but has a width of 1.0. Loss must be influenced by uncertainty_weight.
    prediction_uncertain = jnp.array([[0.0, 1.0]])
    target_mid = jnp.array([[0.5, 0.5]])
    loss_uncertain = jlnn_learning_loss(prediction_uncertain, target_mid, uncertainty_weight=0.1)
    # MSE part: ((0-0.5)^2 + (1-0.5)^2)/2 = 0.25
    # Uncertainty part: 0.1 * (1.0 - 0.0) = 0.1
    # Expected: 0.35
    assert jnp.isclose(loss_uncertain, 0.35)

    # Case: Contradiction check
    # Prediction [0.8, 0.2] is contradictory. 
    # Even if target is [0.5, 0.5], the contradiction_weight should push loss higher.
    prediction_contra = jnp.array([[0.8, 0.2]])
    loss_contra = jlnn_learning_loss(prediction_contra, target_mid, contradiction_weight=10.0)
    assert loss_contra > 1.0, "High contradiction weight should significantly increase loss"

def test_jlnn_learning_loss_gradient_flow():
    """
    Checks if jlnn_learning_loss provides gradients for the uncertainty component.
    This is critical for the 'decisiveness' part of the learning.
    """
    import jax

    def simple_model_output(width):
        # Simulates a model output with a fixed center at 0.5 and variable width
        return jnp.array([[0.5 - width/2, 0.5 + width/2]])

    target = jnp.array([[0.5, 0.5]])
    
    # Calculate gradient of loss with respect to width
    grad_fn = jax.grad(lambda w: jlnn_learning_loss(simple_model_output(w), target, uncertainty_weight=1.0))
    
    # Positive width should produce a gradient that wants to decrease the width
    gradient = grad_fn(0.4)
    assert gradient > 0, "Gradient should be positive to decrease uncertainty (width)"