#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.core import intervals


def contradiction_loss(interval: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the penalty for a logical contradiction in the truth interval.
    
    In Logical Neural Networks (LNN), the axiom of logical consistency must be observed, 
    where the lower bound (L) never exceeds the upper bound (U). If L > U, 
    this means that the system is contradictory 
    (e.g., it simultaneously claims that a statement is 'certainly true' and 'certainly false').
    
    This function calculates the squared loss from the difference between the boundaries 
    if this condition is violated. This motivates the optimizer to adjust the weights 
    and biases to bring the network back into a valid logical space.
    
    Args:
        interval (jnp.ndarray): A tensor of intervals of the form (..., 2). 
            The last dimension contains the pair [Lower Bound, Upper Bound].
                            
    Returns:
        jnp.ndarray: A scalar value representing the average contradiction loss.
    """
    # Extracting boundaries using the core module
    lower = intervals.get_lower(interval)
    upper = intervals.get_upper(interval)
    
    # Calculate the difference (only positive values ​​indicate a dispute)
    # diff = max(0, L - U)
    diff = jnp.maximum(0.0, lower - upper)
    
    # Quadratic penalty for smooth gradient
    return jnp.mean(jnp.square(diff))


def logical_mse_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the mean square error (MSE) between the predicted and target intervals.
    
    This function measures the accuracy of the model by comparing predicted truth intervals with reference values ​​(labels). 
    The calculation is performed simultaneously for both the lower (L) and upper (U) bounds, 
    forcing the model to converge to the target in both truth dimensions.
    
    In the context of JLNN, this loss penalizes deviation from the known truth, 
    while additional functions (such as contradiction_loss) ensure that 
    the resulting gradient move does not violate logical axioms.
    
    Args:
        prediction (jnp.ndarray): Predicted interval tensor of the form (..., 2).
        target (jnp.ndarray): The target (ground truth) tensor of intervals of the form (..., 2).
    Returns:
        jnp.ndarray: A scalar value representing the mean square error.
    """
    # Standard MSE applied to the difference between interval tensors.
    # JAX will automatically broadcast and calculate across both boundaries [L, U].
    return jnp.mean(jnp.square(prediction - target))

def total_lnn_loss(prediction: jnp.ndarray, target: jnp.ndarray, contradiction_weight: float = 1.0) -> jnp.ndarray:
    """
    Calculates the combined loss function (Total Loss) for JLNN.

    This function unifies two main optimization goals in Logical Neural Networks:
        1. Accuracy: Minimizing the difference between the predicted interval and the target using MSE.
        2. Consistency: Penalizing internal inconsistencies (L > U) that would impair the interpretability of the model.

    The resulting gradient value leads the model to find parameters that
    not only describe the data well, but also form a logically closed and consistent system
    in accordance with the axioms of Łukasiewicz's logic.

    Args:
        prediction (jnp.ndarray): The model's output tensor (intervals) of the form (..., 2).
        target (jnp.ndarray): Reference truth values ​​(labels) of the form (..., 2).
        contradiction_weight (float): Hyperparameter determining the strength of the penalty for logical contradiction.
            A higher value (e.g. > 1.0) places more emphasis on logical purity of the model, 
            at the cost of slower MSE error reduction. The default value is 1.0.

    Returns:
        jnp.ndarray: Total scalar loss prepared for gradient calculation in JAX.
    """
    # Calculating the prediction error against the training data
    mse = logical_mse_loss(prediction, target)
    
    # Calculation of penalty for logical contradictions (L > U)
    contra = contradiction_loss(prediction)
    
    # Weighted sum of loss components
    return mse + contradiction_weight * contra


def logical_consistency_loss(model_output: jnp.ndarray, uncertainty_weight: float = 0.1) -> jnp.ndarray:
    """
    A complex loss function to enforce logical consistency and model certainty.

    This function combines two aspects:
    1. **Validity (Hinge Loss)**: Penalizes situations where the lower bound (L) exceeds the upper bound (U). 
    In correct LNN logic, L <= U must always hold.
    2. **Certainty (Uncertainty)**: Minimizes the width of the interval (U - L). Encourages the model to move away from a neutral state of "don't know" (0, 1) towards a definitive "true" (1, 1) or "false" (0, 0).
    so that it does not remain in the neutral state "I don't know" (0, 1), but tends towards "true" (1, 1) or "false" (0, 0).

    Args:
        model_output (jnp.ndarray): The model's output tensor with intervals of the form (..., 2).
        uncertainty_weight (float): Coefficient determining the strength of the pressure to reduce uncertainty. 
            Default value 0.1 ensures that validity and accuracy remain the primary goals.

    Returns:
        jnp.ndarray: Scalar value representing the total inconsistency.
    """
    # Decomposition into lower and upper bounds using the core module
    l = intervals.get_lower(model_output)
    u = intervals.get_upper(model_output)

    # Penalization for 'crossing' (L > U). If L <= U, the value is 0.
    violation = jnp.mean(jnp.maximum(0.0, l - u))

    # Penalization for uncertainty (interval width). We want the bounds to approach each other.
    uncertainty = jnp.mean(u - l)
    
    return violation + uncertainty_weight * uncertainty


def rule_violation_loss(antecedent: jnp.ndarray, consequent: jnp.ndarray) -> jnp.ndarray:
    """
    Penalizes violation of the semantics of logical implication (A -> B).

    In neuro-symbolic learning, this is a key function for knowledge embedding. 
    If a model claims that premise (A) is true (high lower bound) 
    but at the same time claims that conclusion (B) 
    is false (low upper bound), a logical conflict arises that penalizes this function.

    This loss is defined as max(0, L(A) - U(B)).

    Args:
        antecedent (jnp.ndarray): Truth interval of the premise (A).
        consequent (jnp.ndarray): Truth interval of the conclusion (B).

    Returns:
        jnp.ndarray: Average rule violation rate across the batch.
    """
    # L(A) represents the degree to which we are certain of the truth of the assumption
    l_a = intervals.get_lower(antecedent)
    # U(B) represents the maximum possible degree of truth of the conclusion
    u_b = intervals.get_upper(consequent)
    
    # If L(A) > U(B), the rule is violated (the premise is more true than the conclusion can be)
    return jnp.mean(jnp.maximum(0.0, l_a - u_b))


