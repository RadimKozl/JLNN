#!/usr/bin/env python3

# Imports 
import jax
import jax.numpy as jnp
from jlnn.core import intervals, activations

# Numerická tolerance pro bezpečné porovnávání hraničních stavů v float32
EPSILON = 1e-6

# =====================================================================
# 1. ŁUKASIEWICZ'S LOGIC (Nilpotent / Accumulative)
# =====================================================================

def weighted_and_lukasiewicz(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates a weighted Łukasiewicz conjunction (AND) over truth intervals.

    In LNN logic, conjunction is defined through "negative evidence". 
    The closer the inputs are to falsehood (1 - x) and the higher their weight, 
    the more they reduce the overall truth of the result. 
    The parameter beta determines the threshold below which the result is considered absolutely false (0.0).

    Interval semantics:
    Because negation inverts limits, to calculate the upper limit of the result (U) 
    we use the negation of the lower limits of the inputs (L), 
    and to calculate the lower limit of the result (L) 
    we use the negation of the upper limits of the inputs (U).

    Args:
        x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).
        weights (jnp.ndarray): A tensor of weights of the form (num_inputs,).
        beta (jnp.ndarray): Scalar threshold parameter (bias).

    Returns:
        jnp.ndarray: The resulting truth interval [L, U].
    """
    # 1. Calculation of the sum of weighted negations (resistance) for both limits
    # sum_l corresponds to sum(w * (1 - L_input)) -> affects the upper bound of the result
    sum_l = jnp.sum(weights * (1.0 - intervals.get_lower(x)), axis=-1)
    # sum_u equals sum(w * (1 - U_input)) -> affects the lower bound of the result
    sum_u = jnp.sum(weights * (1.0 - intervals.get_upper(x)), axis=-1)
    
    # 2. Application specific AND activation
    # Res_u uses sum_l because less "certain truth" in the input limits the maximum truth of the output.
    res_u = activations.lukasiewicz_and_activation(sum_l, beta)
    res_l = activations.lukasiewicz_and_activation(sum_u, beta)
    
    return intervals.create_interval(res_l, res_u)


def weighted_or_lukasiewicz(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the weighted Łukasiewicz disjunction (OR) over truth intervals.

    Disjunction in LNN acts as an accumulator of "positive evidence". 
    Each true input increases the overall truth of the result depending on its weight. 
    The beta parameter determines how much "evidence" is needed to reach absolute truth (1.0).

    Interval semantics:
    The OR operation preserves the orientation of the limits: the lower limits of the inputs determine 
    the lower limit of the result, and the upper limits of the inputs determine the upper limit of the result.

    Args:
        x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).
        weights (jnp.ndarray): A tensor of weights of the form (num_inputs,).
        beta (jnp.ndarray): Scalar threshold parameter (bias).

    Returns:
        jnp.ndarray: The resulting truth interval [L, U].
    """
    # 1. Weighted sum of confirmed truth
    sum_l = jnp.sum(weights * intervals.get_lower(x), axis=-1)
    sum_u = jnp.sum(weights * intervals.get_upper(x), axis=-1)
    
    # 2. Application of specific OR activation (t-conorm)
    res_l = activations.lukasiewicz_or_activation(sum_l, beta)
    res_u = activations.lukasiewicz_or_activation(sum_u, beta)
    
    return intervals.create_interval(res_l, res_u)

def implies_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Logical implication A -> B (S-implication) based on Łukasiewicz logic.
    
    In JLNN, implication is implemented using logical equivalence:
    (A -> B) ≡ (¬A ∨ B).
    
    This implementation uses interval arithmetic, 
    where negation (NOT) inverts the interval boundaries: NOT [L, U] = [1 - U, 1 - L]. 
    The result is then processed by a weighted OR operator, 
    allowing the model to learn the relevance of a given rule.

    Args:
        int_a (jnp.ndarray): Tensor for the antecedent (presupposition A) of the form (..., 2). 
            The last dimension contains [Lower Bound, Upper Bound].
        int_b (jnp.ndarray): Tensor for the consequent (consequent B) of the form (..., 2). 
            The last dimension contains [Lower Bound, Upper Bound].
        weights (jnp.ndarray): Tensor of weights for an OR gate of the form (2,). 
            The first weight is applied to ¬A, the second to B. Typically initialized to [1, 1].
        beta (float): Threshold (bias) determining the stringency of the implication activation.

    Returns:
        jnp.ndarray: The resulting truth interval of the implication [L, U] of the form (..., 2).
    """
    
    
    # Apply NOT operator in interval logic: [1 - U, 1 - L]
    not_a_L = 1.0 - intervals.get_upper(int_a)
    not_a_U = 1.0 - intervals.get_lower(int_a)
    not_a = intervals.create_interval(not_a_L, not_a_U)
    
    # Pack NOT A and B into a cohesive tensor for bulk processing inside the OR operator
    # axis=-2 inserts a dedicated argument dimension right before the interval dimension
    combined = jnp.stack([not_a, int_b], axis=-2)
    
    # Delegate execution to the previously established weighted OR operator
    return weighted_or_lukasiewicz(combined, weights, beta)

# =====================================================================
# 2. PRODUCT LOGIC (Smooth / Probabilistic)
# =====================================================================

def and_product_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure product conjunction (AND) for two input intervals: A * B."""
    L_res = intervals.get_lower(int_a) * intervals.get_lower(int_b)
    U_res = intervals.get_upper(int_a) * intervals.get_upper(int_b)
    return intervals.create_interval(L_res, U_res)


def or_product_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure product disjunction (OR) for two input intervals: A + B - (A * B)."""
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    L_res = L_a + L_b - (L_a * L_b)
    U_res = U_a + U_b - (U_a * U_b)
    return intervals.create_interval(jnp.clip(L_res, 0.0, 1.0), jnp.clip(U_res, 0.0, 1.0))


def bulk_and_product_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Executes a bulk product AND reduction across the last axis for n-inputs."""
    L_res = jnp.prod(intervals.get_lower(x), axis=-1)
    U_res = jnp.prod(intervals.get_upper(x), axis=-1)
    return intervals.create_interval(L_res, U_res)


def bulk_or_product_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Executes a bulk product OR reduction across the last axis for n-inputs."""
    L_res = 1.0 - jnp.prod(1.0 - intervals.get_lower(x), axis=-1)
    U_res = 1.0 - jnp.prod(1.0 - intervals.get_upper(x), axis=-1)
    return intervals.create_interval(L_res, U_res)


# --- Reichenbach Logic (Compromise / Productive) ---
def implies_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Implements Reichenbach implication (product logic).
    
    This method of computing the implication (A -> B) is defined by the relation 1 - A + (A * B). 
    In the context of interval logic, JLNN represents a "compromise" approach that, 
    unlike Łukasiewicz or Kleene-Dienes, does not contain sharp breaks caused by min/max operations.
    
    The main advantage of this implication is that it is fully differentiable over the entire range [0, 1], 
    which ensures stable and non-zero gradients for both arguments (A and B) simultaneously.

    Args:
        int_a (jnp.ndarray): Input interval for antecedent (assumption) of the form (..., 2).
        int_b (jnp.ndarray): Input interval for the consequent of the form (..., 2).

    Returns:
        jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2). 
            The calculation is performed with respect to interval arithmetic:
                L_res = 1 - U_a + (L_a * L_b)
                U_res = 1 - L_a + (U_a * U_b)
                The result is treated with the clip function to keep the values ​​in the range [0, 1].
    """    
    
    # Extract limits for both input intervals
    L_a = intervals.get_lower(int_a)
    U_a = intervals.get_upper(int_a)
    L_b = intervals.get_lower(int_b)
    U_b = intervals.get_upper(int_b)
    
    # Interval calculation for truth values
    # This form ensures that the lower bound of the result is derived
    # from the most pessimistic combination of inputs and vice versa.
    res_L = 1.0 - U_a + (L_a * L_b)
    res_U = 1.0 - L_a + (U_a * U_b)
    
    # Return the normalized interval
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_goguen(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the pure Goguen implication (Residuation R-implication).
    
    Defined as: 1.0 if A <= B, otherwise B / A.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Safe division in JAX to avoid NaN on backward pass (A -> 0)
    def safe_div(num, denom):
        return jnp.where(denom > 0.0, num / jnp.maximum(denom, 1e-12), 1.0)
    
    # For the lower bound (L): the most pessimistic estimate is the ratio of the smallest B to the largest A
    L_div = safe_div(L_b, U_a)
    res_L = jnp.where(U_a <= L_b, 1.0, L_div)
    
    # Pro horní hranici (U): nejoptimističtější odhad
    U_div = safe_div(U_b, L_a)
    res_U = jnp.where(L_a <= U_b, 1.0, U_div)
    
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))

# =====================================================================
# 3. GÖDEL'S LOGIC (Strict / Extremes / Min-Max)
# =====================================================================

def and_godel_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure Godel conjunction (AND): min(A, B)."""
    L_res = jnp.minimum(intervals.get_lower(int_a), intervals.get_lower(int_b))
    U_res = jnp.minimum(intervals.get_upper(int_a), intervals.get_upper(int_b))
    return intervals.create_interval(L_res, U_res)


def or_godel_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure Godel disjunction (OR): max(A, B)."""
    L_res = jnp.maximum(intervals.get_lower(int_a), intervals.get_lower(int_b))
    U_res = jnp.maximum(intervals.get_upper(int_a), intervals.get_upper(int_b))
    return intervals.create_interval(L_res, U_res)


def bulk_and_godel_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure Godel AND (min reduction) along the designated last axis."""
    return intervals.create_interval(jnp.min(intervals.get_lower(x), axis=-1), jnp.min(intervals.get_upper(x), axis=-1))


def bulk_or_godel_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure Godel OR (max reduction) along the designated last axis."""
    return intervals.create_interval(jnp.max(intervals.get_lower(x), axis=-1), jnp.max(intervals.get_upper(x), axis=-1))


# --- Kleene-Dienes Logic (Pessimistic/Extreme) ---

def implies_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Implements Kleene-Dienes implication (standard max-min logic).
    
    This method of computing the implication (A -> B) is defined as max(1 - A, B). 
    In the context of interval logic JLNN, this is a "pessimistic" approach, 
    since the resulting truth depends only on the most significant extreme 
    (either the antecedent is false or the consequent is true).
    
    Unlike Łukasiewicz logic, there is no linear addition of truths, 
    which can be useful for robust systems resistant to the accumulation of small errors.

    Args:
        int_a (jnp.ndarray): Input interval for antecedent (assumption) of the form (..., 2).
        int_b (jnp.ndarray): Input interval for the consequent of the form (..., 2).
    Returns:
        jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2). 
            The calculation is as follows:
                L_res = max(1 - U_a, L_b)
                U_res = max(1 - L_a, U_b)
    """
    
    # Calculating the negation of the antecedent (NOT A) in interval arithmetic    
    not_a_L = 1.0 - intervals.get_upper(int_a)
    not_a_U = 1.0 - intervals.get_lower(int_a)
    
    # Obtaining the bounds of the consequent (B)
    L_b = intervals.get_lower(int_b)
    U_b = intervals.get_upper(int_b)
    
    # Kleene-Dienes takes the maximum of (NOT A) and (B)
    res_L = jnp.maximum(not_a_L, L_b)
    res_U = jnp.maximum(not_a_U, U_b)
    
    return intervals.create_interval(res_L, res_U)


def implies_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure Godel implication (R-implication residuum).
    
    Defined as: 1.0 if A <= B, otherwise B.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    res_L = jnp.where(U_a <= L_b, 1.0, L_b)
    res_U = jnp.where(L_a <= U_b, 1.0, U_b)
    return intervals.create_interval(res_L, res_U)


# =====================================================================
# 4. DRASTIC LOGIC (Lower Theoretical Barrier / Drastic Product)
# =====================================================================

def and_drastic_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Pure drastic t-norm (AND). Represents the absolute lower bound of all valid t-norms.
    
    Returns the opposite argument if one equals 1.0, otherwise collapses to 0.0.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    res_L = jnp.where(L_a >= 1.0 - EPSILON, L_b, jnp.where(L_b >= 1.0 - EPSILON, L_a, 0.0))
    res_U = jnp.where(U_a >= 1.0 - EPSILON, U_b, jnp.where(U_b >= 1.0 - EPSILON, U_a, 0.0))
    return intervals.create_interval(res_L, res_U)


def or_drastic_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Pure drastic t-conorm (OR). Represents the absolute upper bound of all valid t-conorms.
    
    Returns the opposite argument if one equals 0.0, otherwise saturates to 1.0.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    res_L = jnp.where(L_a <= EPSILON, L_b, jnp.where(L_b <= EPSILON, L_a, 1.0))
    res_U = jnp.where(U_a <= EPSILON, U_b, jnp.where(U_b <= EPSILON, U_a, 1.0))
    return intervals.create_interval(res_L, res_U)


def bulk_and_drastic_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure drastic AND reduction along the designated last axis."""
    L = intervals.get_lower(x)
    U = intervals.get_upper(x)
    
    # Safe masks only sharp '!= 1.0'
    not_one_L = jnp.sum(L < 1.0 - EPSILON, axis=-1)
    not_one_U = jnp.sum(U < 1.0 - EPSILON, axis=-1)
    
    # Drastic reduction: if all are 1 -> 1. If exactly one is < 1 -> return the minimum element. Otherwise 0.
    min_L = jnp.min(L, axis=-1)
    min_U = jnp.min(U, axis=-1)
    
    res_L = jnp.where(not_one_L == 0, 1.0, jnp.where(not_one_L == 1, min_L, 0.0))
    res_U = jnp.where(not_one_U == 0, 1.0, jnp.where(not_one_U == 1, min_U, 0.0))
    return intervals.create_interval(res_L, res_U)


def bulk_or_drastic_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure drastic OR reduction along the designated last axis."""
    L = intervals.get_lower(x)
    U = intervals.get_upper(x)
    
    # Safe masks only sharp '!= 0.0'
    not_zero_L = jnp.sum(L > EPSILON, axis=-1)
    not_zero_U = jnp.sum(U > EPSILON, axis=-1)
    
    max_L = jnp.max(L, axis=-1)
    max_U = jnp.max(U, axis=-1)
    
    res_L = jnp.where(not_zero_L == 0, 0.0, jnp.where(not_zero_L == 1, max_L, 1.0))
    res_U = jnp.where(not_zero_U == 0, 0.0, jnp.where(not_zero_U == 1, max_U, 1.0))
    return intervals.create_interval(res_L, res_U)


# =====================================================================
# 5. Space-Curved Physical Fuzzy Logic (PFL)
# =====================================================================

# ---------------------------------------------------------------------
# PHYSICAL IMPLICATIONS (PFL IMPLICATIONS) OVER INTERVALS
# ---------------------------------------------------------------------

def implies_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Physical Kleene-Dienes implication modulated by localized entropic chaos.
    
    Mathematical formula:
    max(1 - (1 - H(A)) * A, (1 - H(B)) * B)
    
    Applied independently on interval bounds conforming to pessimistic interval semantics.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Calculation of stabilizing weights (1 - H)
    # For the negation of the antecedent (not A) in intervals, the bounds are inverted:
    # The lower bound depends on U_a, the upper bound depends on L_a
    w_U_a = activations.get_entropic_weight(U_a)
    w_L_a = activations.get_entropic_weight(L_a)
    w_L_b = activations.get_entropic_weight(L_b)
    w_U_b = activations.get_entropic_weight(U_b)
    
    # Synthesize the localized physical "NOT A" and physical "B" expressions
    not_a_L = 1.0 - (w_U_a * U_a)
    not_a_U = 1.0 - (w_L_a * L_a)
    
    b_weighted_L = w_L_b * L_b
    b_weighted_U = w_U_b * U_b
    
    # Execute the traditional Kleene-Dienes MAX operator over the deformed physical truth field
    res_L = jnp.maximum(not_a_L, b_weighted_L)
    res_U = jnp.maximum(not_a_U, b_weighted_U)
    
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Physical Reichenbach implication via a smooth gravitational polynomial.
    
    Mathematical formula:
    1 - A + A * B * (1 - H(A) * H(B))
    
    In states of maximum system chaos (where both inputs converge near 0.5), the 
    interactive coupling term collapses. This opens a clear linear gradient channel 
    to facilitate uninterrupted error backpropagation.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Extract structural entropy fields from individual component boundaries
    h_L_a, h_U_a = activations.entropy_raw(L_a), activations.entropy_raw(U_a)
    h_L_b, h_U_b = activations.entropy_raw(L_b), activations.entropy_raw(U_b)
    
    # Formulate space deformation coefficients governing the interactive product term (A * B)
    # The pessimistic lower bound combines lower bound uncertainty estimations
    deform_L = 1.0 - (h_L_a * h_L_b)
    deform_U = 1.0 - (h_U_a * h_U_b)
    
    # Evaluate interval execution of the Reichenbach polynomial: 1 - A + (A * B * Deform)
    # Lower bound (pessimistic): Minimize truth -> subtract maximal U_a, add minimal product term
    res_L = 1.0 - U_a + (L_a * L_b * deform_L)
    # Upper bound (optimistic): Maximize truth -> subtract minimal L_a, add maximal product term
    res_U = 1.0 - L_a + (U_a * U_b * deform_U)
    
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_physical_gravitational_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Physical Gravitational Lukasiewicz implication.
    
    Mathematical formula:
    min(1.0, 1.0 - (1 - H(A)) * A + (1 - H(B)) * B)
    
    This operator mirrors classical truth values when states exhibit low entropy, 
    while establishing total coherence (truth = 1.0) as inputs approach the entropic singularity [0.5, 0.5].
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Extract functional entropic weights (1.0 - H) across individual boundary bounds
    w_L_a, w_U_a = activations.get_entropic_weight(L_a), activations.get_entropic_weight(U_a)
    w_L_b, w_U_b = activations.get_entropic_weight(L_b), activations.get_entropic_weight(U_b)
    
    # Perform interval evaluation while preserving proper boundary inversion within the negative expression:
    # L_res = 1 - U_weighted_a + L_weighted_b
    # U_res = 1 - L_weighted_a + U_weighted_b
    res_L = 1.0 - (w_U_a * U_a) + (w_L_b * L_b)
    res_U = 1.0 - (w_L_a * L_a) + (w_U_b * U_b)
    
    # Enforce standard Lukasiewicz upper boundary saturation at 1.0
    return intervals.create_interval(jnp.minimum(1.0, res_L), jnp.minimum(1.0, res_U))
