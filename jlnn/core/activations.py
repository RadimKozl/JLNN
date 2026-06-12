#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp


def identity_activation(x: jnp.ndarray) -> jnp.ndarray:
    """
    Realizes the identity function truncated to the closed logical interval [0, 1].
    
    This activation function is utilized within the JLNN framework primarily as a 
    numerical safeguard in locations where inputs already semantically represent 
    truth values (e.g., outputs from upstream predicates or logical gates).
    
    It ensures that minor numerical inaccuracies or floating-point drifts arising 
    from cumulative tensor operations do not leak outside the valid logical boundary. 
    In the context of interval logic, it maintains axiomatic integrity by enforcing 
    strict saturation at 0.0 (absolute falsehood) and 1.0 (absolute truth).
    
    Args:
        x (jnp.ndarray): Input tensor of arbitrary shape containing truth values 
            or raw logical potentials.
        
    Returns:
        jnp.ndarray: A bounded tensor of the same shape as the input, where each 
            element v_i satisfies the axiomatic constraint 0.0 <= v_i <= 1.0.
    """
    
    # Using jnp.clip is highly efficient in JAX as it translates to hardware-native
    # min/max primitives and correctly routes zero-gradients for out-of-bounds regions.
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
    # Clip ensures strict saturation at 0.0 and 1.0, preserving algebraic boundaries 
    # and preventing out-of-bound gradient leakage during the optimization phase.
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
    # Utilizing jnp.clip guarantees strict compliance with the logical unit interval,
    # mitigating minor floating-point drifts during extensive backpropagation.
    return jnp.clip(sum_val / beta, 0.0, 1.0)


def ramp_sigmoid(x: jnp.ndarray, slope: float = 1.0, offset: float = 0.5) -> jnp.ndarray:
    """
    Implements a linear Ramp activation function (truncated linear mapping).

    Within the JLNN framework, this function is primarily utilized in grounding 
    layers (such as LearnedPredicate) to transform unconstrained real-valued input 
    features into fuzzy truth values. It balances the computational benefits of 
    linearity (interpretability and constant gradients) with the semantic benefits 
    of strict boundary saturation.

    Parameter Semantics:
        - offset: Specifies the shift on the X-axis where the truth value is 
          exactly 0.5 (the logical midpoint / decision boundary).
        - slope: Controls the stringency of the predicate. A high slope yields 
          a sharp transition between false and true, approximating a crisp step function.

    Args:
        x (jnp.ndarray): Input tensor of raw real-valued measurements (e.g., physical telemetry).
        slope (float): Steepness coefficient of the linear segment. Controls transition width.
            Defaults to 1.0.
        offset (float): Horizontal translation factor on the X-axis. Sets the decision threshold.
            Defaults to 0.5.

    Returns:
        jnp.ndarray: Grounded fuzzy truth value constrained to the interval [0, 1].
    """
    # The addition of 0.5 ensures that when the input x perfectly matches the offset,
    # the resulting truth value maps precisely to the logical center of uncertainty (0.5).
    return jnp.clip(slope * (x - offset) + 0.5, 0.0, 1.0)


# =====================================================================
# SPACE-CURVED PHYSICAL FUZZY LOGIK (PFL) MODULE EXTENSION
# =====================================================================

# ---------------------------------------------------------------------
# INTERNAL ENTROPIC KERNEL (SHANNON)
# ---------------------------------------------------------------------

def entropy_raw(val: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the normalized binary Shannon entropy over the interval [0, 1].
    
    Acts as an exported framework utility for measuring local fuzzy uncertainty 
    and systemic chaos. The output is normalized such that maximum entropy 
    H(0.5) = 1.0, and deterministic edges H(0.0) = H(1.0) = 0.0.
    
    Args:
        val (jnp.ndarray): Input logical tensor containing truth values nominally in [0, 1].
        
    Returns:
        jnp.ndarray: Computed and normalized Shannon entropy values.
    """
    eps = 1e-7
    # Safe clipping to avoid log(0) and log(negative number)
    v_clipped = jnp.clip(val, eps, 1.0 - eps)
    
    # Shannon entropy formula: -p*log2(p) - (1-p)*log2(1-p)
    h = -v_clipped * jnp.log2(v_clipped) - (1.0 - v_clipped) * jnp.log2(1.0 - v_clipped)
    
    # Numerical safeguard for extreme edges where entropy is analytically zero
    return jnp.where((val <= eps) | (val >= 1.0 - eps), 0.0, h)


def get_entropic_weight(val: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the logical stability (entropic weight) of a state: 1.0 - H(val).
    
    This function is exported for use across interval-valued physical implication 
    operators (PFL) to dynamically adjust logical truth bounds based on local uncertainty.
    
    Args:
        val (jnp.ndarray): Input logical truth value or individual interval bound.
        
    Returns:
        jnp.ndarray: Logical stability weight in the closed range [0, 1].
    """
    return 1.0 - entropy_raw(val)


def gravitational_bend_activation(
    z: jnp.ndarray, 
    gamma: float = 0.2, 
    mode: str = 'sigmoid',
    slope: float = 1.0,
    offset: float = 0.5
) -> jnp.ndarray:
    """
    PFL activation function that deforms the standard truth potential space [0, 1].
    
    Simulates a gravitational well around the logical center (0.5) via local entropy, 
    pulling highly unstable, uncertain states towards the center. Near the deterministic 
    edges (0.0 and 1.0), this gravitational influence decays naturally, allowing the 
    function to converge to standard saturation behavior.
    
    Args:
        z (jnp.ndarray): Input logical potential (raw input features or linear combination).
        gamma (float): Bending strength coefficient, bounded within the interval [0, 1].
            Defaults to 0.2.
        mode (str): Base compression strategy. Options are 'sigmoid' (smooth physical field) 
            or 'ramp' (truncated linear mapping). Defaults to 'sigmoid'.
        slope (float): Stringency parameter utilized exclusively when mode='ramp'. 
            Defaults to 1.0.
        offset (float): Midpoint shift parameter utilized exclusively when mode='ramp'. 
            Defaults to 0.5.
        
    Returns:
        jnp.ndarray: Bounded truth value in [0, 1] after entropic space bending.
    """
    # 1. Compress raw potentials to the base logical interval [0, 1] using the specified mode
    if mode == 'sigmoid':
        # Smooth continuous physical field simulation
        base_truth = 1.0 / (1.0 + jnp.exp(-z))
    elif mode == 'ramp':
        # Sharper transitions with explicit clipping limits
        base_truth = ramp_sigmoid(z, slope=slope, offset=offset)
    else:
        raise ValueError(f"Unknown PFL activation mode: '{mode}'. Choose 'sigmoid' or 'ramp'.")
    
    # 2. Compute local Shannon entropy at the compressed coordinate
    h = entropy_raw(base_truth)
    
    # 3. Calculate the directional restoring gravitational force towards the center (0.5)
    restoring_force = 0.5 - base_truth
    
    # 4. Apply non-linear space deformation proportional to entropy and gamma strength
    a = base_truth + gamma * h * restoring_force
    
    return jnp.clip(a, 0.0, 1.0)