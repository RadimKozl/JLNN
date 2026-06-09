#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp


def identity_activation(x: jnp.ndarray) -> jnp.ndarray:
    """
    Realizes the identity function truncated to the closed interval [0, 1].
    
    This activation function is used in JLNN primarily as a numerical safeguard 
    in places where inputs already semantically represent truth values (e.g., 
    outputs from predicates or downstream logical gates).
    
    It ensures that minor numerical inaccuracies arising from floating-point 
    computations do not leak outside the valid logical boundary. In the context 
    of interval logic, it maintains axiomatic integrity by enforcing strict 
    saturation at 0.0 (absolute falsehood) and 1.0 (absolute truth).
    
    Args:
        x (jnp.ndarray): Input tensor of arbitrary shape containing truth values 
            or raw logical potentials.
        
    Returns:
        jnp.ndarray: A tensor of the same shape as the input, where each 
            element v_i satisfies the constraint 0.0 <= v_i <= 1.0.
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


# =====================================================================
# SPACE-CURVED PHYSICAL FUZZY LOGIK (PFL) MODULE EXTENSION
# =====================================================================

# ---------------------------------------------------------------------
# INTERNAL ENTROPIC KERNEL (SHANNON)
# ---------------------------------------------------------------------

def entropy_raw(val: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the normalized binary Shannon entropy over the interval [0, 1].
    
    Acts as an exported framework utility for measuring local fuzzy uncertainty.
    The output is normalized such that H(0.5) = 1.0 and H(0.0) = H(1.0) = 0.0.
    
    Args:
        val (jnp.ndarray): Input logical tensor containing values nominally in [0, 1].
        
    Returns:
        jnp.ndarray: Computed Shannon entropy values.
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
    operators (PFL) to dynamically adjust truth bounds based on systemic chaos.
    
    Args:
        val (jnp.ndarray): Input logical truth value or interval bound.
        
    Returns:
        jnp.ndarray: Logical stability weight in the range [0, 1].
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
    
    Around the logical center (0.5), it simulates a gravitational well via local entropy, 
    attracting unstable, highly uncertain states. Near the deterministic edges (0, 1), 
    the gravitational influence decays naturally, behaving like standard saturation.
    
    Args:
        z (jnp.ndarray): Input logical potential (linear combination or raw input features).
        gamma (float): Strength of the gravitational bending, bounded in the interval [0, 1].
        mode (str): Base compression method. Options are 'sigmoid' (smooth physical field) 
            or 'ramp' (truncated linear mapping).
        slope (float): Stringency parameter used exclusively if mode='ramp'. Default is 1.0.
        offset (float): Center shift parameter used exclusively if mode='ramp'. Default is 0.5.
        
    Returns:
        jnp.ndarray: Truth value in the interval [0, 1] after gravitational deformation.
    """
    # 1. Base compression to logical interval [0, 1] based on user configuration
    if mode == 'sigmoid':
        # Smooth physical field simulation
        base_truth = 1.0 / (1.0 + jnp.exp(-z))
    elif mode == 'ramp':
        # Sharper transitions with explicit saturation bounds - calling the sibling function directly
        base_truth = ramp_sigmoid(z, slope=slope, offset=offset)
    else:
        raise ValueError(f"Unknown PFL activation mode: '{mode}'. Choose 'sigmoid' or 'ramp'.")
    
    # 2. Calculation of local Shannon entropy at the compressed point using the local public function
    h = entropy_raw(base_truth)
    
    # 3. Application of the restoring gravitational force towards the center (0.5)
    restoring_force = 0.5 - base_truth
    
    # 4. Resulting space bending around the entropic singularity
    a = base_truth + gamma * h * restoring_force
    
    return jnp.clip(a, 0.0, 1.0)