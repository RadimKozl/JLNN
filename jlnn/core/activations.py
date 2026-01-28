#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp


def identity_activation(x: jnp.ndarray) -> jnp.ndarray:
    """
    It realizes the identity truncated to the closed interval [0, 1].
    
    This activation function is used in JLNN primarily as 
    a numerical safeguard in places where inputs already semantically represent truth values 
    ​(e.g., outputs from predicates or other gates).
    
    It ensures that minor numerical inaccuracies arising from floating point calculations 
    do not lead to values ​​outside the valid logical range.
    
    In the context of interval logic, [L, U] helps maintain axiomatic integrity 
    by enforcing saturation at the boundaries of 0.0 (absolute falsehood) and 1.0 (absolute truth).
    
    Args:
        x (jnp.ndarray): Input tensor of arbitrary shape containing truth values ​​or logical potentials.
        
    Returns:
        jnp.ndarray: A tensor of the same shape as the input, where each value v_i satisfies the condition 0.0 <= v_i <= 1.0.
    """
    
    # Using jnp.clip is an efficient operation in JAX that defines
    # a constant zero for out-of-range values ​​in backpropagation.
    return jnp.clip(x, 0.0, 1.0)


def lukasiewicz_and_activation(sum_val: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    It implements the activation function for the weighted Łukasiewicz conjunction (AND).

    This function transforms the weighted sum of the logical negations 
    of the inputs into the resulting truth value. In LNN logic, 
    the beta parameter plays the role of a sensitivity threshold: 
    if the weighted sum of the 'false's' on the inputs exceeds the beta value, 
    the gate output linearly drops to zero.

    Mathematical relationship:
    f(s, β) = max(0, 1 - (s / β))
    where 's' is the weighted sum (sum_val) and 'β' is the threshold (beta).

    Features:
    - For s = 0 (all inputs are 1.0): Output is 1.0 (True).
    - For s >= β: Output is 0.0 (False).
    - It preserves a linear gradient in the region (0, β), which is crucial for learning weights.

    Args:
        sum_val (jnp.ndarray): Weighted sum of the complements of the truth values ​​of the inputs.
            Typically calculated as sum(w * (1 - x)).
        beta (jnp.ndarray): Gate threshold (nnx.Param), determining the steepness
            and the breakpoint of the logic function.

    Returns:
        jnp.ndarray: Resulting truth value in the interval [0, 1].
    """
    # The implementation uses jnp.maximum to implement the lower clipping. 
    # Upper trimming to 1.0 is not necessary for standard ANDs with positive weights, 
    # but ensures stability during training.
    return jnp.clip(1.0 - (sum_val / beta), 0.0, 1.0)


def lukasiewicz_or_activation(sum_val: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    It implements the activation function for the weighted Łukasiewicz disjunction (OR).

    This function transforms the weighted sum of the truth values 
    ​​of the inputs into the resulting truth of the output. 
    In LNN semantics, the beta parameter determines the sensitivity of the gate: 
    the lower the beta value, the fewer "confirmations" (the sum of the inputs) 
    are needed to reach the full truth.

    Mathematical relationship:
    f(s, β) = min(1, s / β)
    where 's' is the weighted sum (sum_val) and 'β' is the threshold (beta).

    Features:
    - For s = 0 (all inputs are 0.0): Output is 0.0 (False).
    - For s >= β: Output is 1.0 (True).
    - The linear increase between 0 and β enables efficient gradient learning of weights and thresholds.

    We use jnp.clip instead of jnp.minimum to ensure strict adherence 
    to the interval [0, 1] even with numerical inaccuracies 
    (e.g. floating-point drift) during the training phase.

    Args:
        sum_val (jnp.ndarray): Weighted sum of the truth values ​​of the inputs.
            Calculated as sum(w * x).
        beta (jnp.ndarray): Gate threshold (nnx.Param), 
            determining the saturation point logical truths.

    Returns:
        jnp.ndarray: Resulting truth value in the interval [0, 1].
    """
    # Using clip ensures saturation to 1.0 and at the same time zeroes the gradient below 0.0,
    # which keeps the model in a logically defined space.
    return jnp.clip(sum_val / beta, 0.0, 1.0)


def ramp_sigmoid(x: jnp.ndarray, slope: float = 1.0, offset: float = 0.5) -> jnp.ndarray:
    """
    It implements a linear "Ramp" activation (truncated linear function).

    Within JLNN, this function is used in predicates (LearnedPredicate) 
    to convert real input values ​​to logical truth values. 
    It combines the advantages of linearity (interpretability and stable gradient) 
    with the advantages of saturation (clear delineation of absolute truth and falsehood).

    Parameter semantics:
    - offset: Specifies the point on the X-axis where the truth is exactly 0.5 (midpoint).
    - slope: Determines the "stringency" of the predicate. 
    A high slope means a fast transition between false and true (close to a step function).

    Args:
        x (jnp.ndarray): Input tensor of real numbers (e.g. temperature, pressure, distance).
        slope (float): Slope of the linear part. Affects the width of the transition area. 
            Default value is 1.0.
        offset (float): Shifts the function on the X-axis. Determines the decision boundary. 
            Default value is 0.5.

    Returns:
        jnp.ndarray: Truth value in the interval [0, 1].
    """
    # Calculation of a linear relationship followed by clipping.
    # The value 0.5 ensures that when x = offset the result is exactly the middle of the logical interval.
    return jnp.clip(slope * (x - offset) + 0.5, 0.0, 1.0)