#!/usr/bin/env python3

# Imports 
import jax
import jax.numpy as jnp
from jlnn.core import intervals, activations

# Numerical tolerance for secure boundary state comparisons in float32 precision
EPSILON = 1e-6

# =====================================================================
# 1. ŁUKASIEWICZ'S LOGIC (Nilpotent / Accumulative)
# =====================================================================

def and_lukasiewicz_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure, parameterless Łukasiewicz conjunction (t-norm) for two intervals.
    
    Mathematical formula: max(0.0, A + B - 1.0) evaluated over interval boundaries.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Nilpotent bounded difference interaction
    res_L = jnp.maximum(0.0, L_a + L_b - 1.0)
    res_U = jnp.maximum(0.0, U_a + U_b - 1.0)
    return intervals.create_interval(res_L, res_U)


def or_lukasiewicz_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure, parameterless Łukasiewicz disjunction (t-conorm) for two intervals.
    
    Mathematical formula: min(1.0, A + B) evaluated over interval boundaries.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Nilpotent bounded sum interaction
    res_L = jnp.minimum(1.0, L_a + L_b)
    res_U = jnp.minimum(1.0, U_a + U_b)
    return intervals.create_interval(res_L, res_U)


def weighted_and_lukasiewicz(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates a weighted Łukasiewicz conjunction (AND) over truth intervals.

    In LNN (Logical Neural Network) logic, conjunction is structurally formulated 
    through "negative evidence". The closer the inputs gravitate toward falsehood (1 - x) 
    and the higher their respective importance weight, the more aggressively they 
    suppress the overall truth value of the result. The parameter beta governs 
    the structural sensitivity threshold below which the output completely 
    saturates to absolute falsehood (0.0).

    Interval Semantics:
    Because negation algebraically inverts boundary limits, computing the upper limit 
    of the result (U) requires evaluating the negated lower limits of the inputs (L). 
    Conversely, computing the lower limit of the result (L) requires evaluating 
    the negated upper limits of the inputs (U).

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): A tensor of input importance weights structured as (num_inputs,).
        beta (jnp.ndarray): Scalar activation sensitivity threshold parameter (bias).

    Returns:
        jnp.ndarray: The resulting bounded truth interval [L, U] structured as (..., 2).
    """
    # 1. Compute the cumulative sum of weighted negations (logical resistance) for both boundaries
    # sum_l aggregates w_i * (1 - L_input), directly constraining the upper bound of the result
    sum_l = jnp.sum(weights * (1.0 - intervals.get_lower(x)), axis=-1)
    # sum_u aggregates w_i * (1 - U_input), directly constraining the lower bound of the result
    sum_u = jnp.sum(weights * (1.0 - intervals.get_upper(x)), axis=-1)
    
    # 2. Apply the specialized Łukasiewicz AND activation functions
    # res_u maps sum_l because a lack of absolute truth at the lower input limits caps maximum potential output truth
    res_u = activations.lukasiewicz_and_activation(sum_l, beta)
    res_l = activations.lukasiewicz_and_activation(sum_u, beta)
    
    return intervals.create_interval(res_l, res_u)


def weighted_or_lukasiewicz(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the weighted Łukasiewicz disjunction (OR) over truth intervals.

    Disjunction within the LNN framework acts as an explicit accumulator of "positive evidence". 
    Each validating input signal increments the overall cumulative truth of the outcome proportional 
    to its assigned weight. The beta threshold parameter regulates how much aggregate positive validation 
    is required to establish absolute saturation at perfect truth (1.0).

    Interval Semantics:
    The OR operation preserves standard boundary orientation: the lower limits of the inputs 
    uniquely determine the lower limit of the result, while the upper limits of the inputs 
    determine the corresponding upper limit of the result.

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): A tensor of input importance weights structured as (num_inputs,).
        beta (jnp.ndarray): Scalar activation saturation threshold parameter (bias).

    Returns:
        jnp.ndarray: The resulting bounded truth interval [L, U] structured as (..., 2).
    """
    # 1. Accumulate the weighted sum of positive validation across boundaries
    sum_l = jnp.sum(weights * intervals.get_lower(x), axis=-1)
    sum_u = jnp.sum(weights * intervals.get_upper(x), axis=-1)
    
    # 2. Map cumulative evidence using the specialized Łukasiewicz OR t-conorm activation
    res_l = activations.lukasiewicz_or_activation(sum_l, beta)
    res_u = activations.lukasiewicz_or_activation(sum_u, beta)
    
    return intervals.create_interval(res_l, res_u)

def implies_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Computes the logical implication A -> B (S-implication) based on Łukasiewicz logic.
    
    Within the JLNN framework, implication is modeled using classical logical equivalence:
        (A -> B) identical to (NOT A OR B).
    
    This routine utilizes interval arithmetic, where logical negation (NOT) inverts 
    the boundaries: NOT [L, U] = [1 - U, 1 - L]. The resulting expressions are then processed 
    by a parameterized weighted OR operator, enabling the neural architecture to learn 
    the optimization weights and structural relevance of expert-defined rules.

    Args:
        int_a (jnp.ndarray): Antecedent tensor (premise A) structured as (..., 2), 
            where the final dimension holds [Lower Bound, Upper Bound].
        int_b (jnp.ndarray): Consequent tensor (conclusion B) structured as (..., 2), 
            where the final dimension holds [Lower Bound, Upper Bound].
        weights (jnp.ndarray): Optimization weight tensor for the underlying OR gate structured as (2,). 
            The first component scales NOT A, the second scales B. Typically initialized to [1.0, 1.0].
        beta (float): Threshold parameter (bias) establishing the activation stringency of the implication.

    Returns:
        jnp.ndarray: The resulting truth interval of the implication [L, U] structured as (..., 2).
    """
    
    # Apply the NOT operator under interval arithmetic constraints: [1 - U, 1 - L]
    not_a_L = 1.0 - intervals.get_upper(int_a)
    not_a_U = 1.0 - intervals.get_lower(int_a)
    not_a = intervals.create_interval(not_a_L, not_a_U)
    
    # Package NOT A and B into a unified tensor for batch calculation inside the OR operator.
    # axis=-2 introduces a dedicated argument dimension immediately preceding the boundary dimension.
    combined = jnp.stack([not_a, int_b], axis=-2)
    
    # Delegate structural processing to the established weighted Łukasiewicz OR operator
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
    Implements the Reichenbach implication (product logic s-implication).
    
    This method of computing implication (A -> B) is governed by the polynomial expression:
    1.0 - A + (A * B)
     
    In the context of interval logic, this represents a smooth "compromise" approach that, 
    unlike Łukasiewicz or Kleene-Dienes, avoids sharp optimization boundaries caused by min/max operations.
    
    The primary mathematical advantage is that it is fully differentiable across the entire 
    unit domain [0, 1], guaranteeing stable, non-vanishing gradients for both antecedent and 
    consequent parameters simultaneously.

    Args:
        int_a (jnp.ndarray): Input interval for the antecedent structured as (..., 2).
        int_b (jnp.ndarray): Input interval for the consequent structured as (..., 2).

    Returns:
        jnp.ndarray: The resulting truth interval [L, U] structured as (..., 2). 
        
        The boundary calculation satisfies rigorous interval arithmetic constraints:
        
        L_res = 1.0 - U_a + (L_a * L_b)
        U_res = 1.0 - L_a + (U_a * U_b)
        
        Outputs are bounded using jnp.clip to shield downstream layers from floating-point overflow.
    """
    
    # Extract boundary limits for both input intervals
    L_a = intervals.get_lower(int_a)
    U_a = intervals.get_upper(int_a)
    L_b = intervals.get_lower(int_b)
    U_b = intervals.get_upper(int_b)
    
    # Execute boundary-safe interval operations
    # This combination ensures that the lower output limit is derived from the most pessimistic
    # algebraic configuration of the inputs, while the upper limit captures the most optimistic setting.
    res_L = 1.0 - U_a + (L_a * L_b)
    res_U = 1.0 - L_a + (U_a * U_b)
    
    # Return the normalized interval
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_goguen(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the pure Goguen implication (Residuation R-implication).
    
    Defined algebraically as: 1.0 if A <= B, otherwise B / A.
    
    Args:
        int_a (jnp.ndarray): Antecedent truth interval.
        int_b (jnp.ndarray): Consequent truth interval.
        
    Returns:
        jnp.ndarray: Bounded truth interval evaluated via Goguen residuation.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Safe numerical division primitive in JAX to eliminate NaN anomalies on backpropagation paths (A -> 0.0)
    def safe_div(num, denom):
        return jnp.where(denom > 0.0, num / jnp.maximum(denom, 1e-12), 1.0)
    
    # For the lower bound (L): the most pessimistic estimate relies on the ratio of minimum B to maximum A
    L_div = safe_div(L_b, U_a)
    res_L = jnp.where(U_a <= L_b, 1.0, L_div)
    
    # For the upper bound (U): the most optimistic estimate relies on the ratio of maximum B to minimum A
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
    Implements the Kleene-Dienes implication (standard max-min logical model).
    
    This implication strategy (A -> B) is explicitly defined as max(1.0 - A, B). 
    In the context of JLNN interval logic, this represents a highly "pessimistic" approach, 
    given that the resulting truth depends entirely on the dominant active extreme 
    (either the complete falsity of the premise or the total truth of the conclusion).
    
    Unlike Łukasiewicz models, it avoids linear addition of truths, making it highly 
    suitable for structural systems designed to be robust against cumulative edge errors.

    Args:
        int_a (jnp.ndarray): Input interval for the antecedent structured as (..., 2).
        int_b (jnp.ndarray): Input interval for the consequent structured as (..., 2).
        
    Returns:
        jnp.ndarray: The resulting truth interval [L, U] structured as (..., 2).
            Algebraic boundary configurations map as follows:
                L_res = max(1.0 - U_a, L_b)
                U_res = max(1.0 - L_a, U_b)
    """
    
    # Evaluate negation of the antecedent (NOT A) under strict interval logic rules   
    not_a_L = 1.0 - intervals.get_upper(int_a)
    not_a_U = 1.0 - intervals.get_lower(int_a)
    
    # Retrieve boundary values of the conclusion (B)
    L_b = intervals.get_lower(int_b)
    U_b = intervals.get_upper(int_b)
    
    # Resolve the final Kleene-Dienes state using parallel JAX maximum primitives
    res_L = jnp.maximum(not_a_L, L_b)
    res_U = jnp.maximum(not_a_U, U_b)
    
    return intervals.create_interval(res_L, res_U)


def implies_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure Gödel implication (R-implication residuum).
    
    Defined algebraically as: 1.0 if A <= B, otherwise B.
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
    Pure drastic t-norm (AND). Represents the absolute mathematical lower bound of all valid t-norms.
    
    Returns the opposing argument if one component equals exactly 1.0, otherwise collapses completely to 0.0.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    res_L = jnp.where(L_a >= 1.0 - EPSILON, L_b, jnp.where(L_b >= 1.0 - EPSILON, L_a, 0.0))
    res_U = jnp.where(U_a >= 1.0 - EPSILON, U_b, jnp.where(U_b >= 1.0 - EPSILON, U_a, 0.0))
    return intervals.create_interval(res_L, res_U)


def or_drastic_pure(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Pure drastic t-conorm (OR). Represents the absolute mathematical upper bound of all valid t-conorms.
    
    Returns the opposing argument if one component equals exactly 0.0, otherwise saturates completely to 1.0.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    res_L = jnp.where(L_a <= EPSILON, L_b, jnp.where(L_b <= EPSILON, L_a, 1.0))
    res_U = jnp.where(U_a <= EPSILON, U_b, jnp.where(U_b <= EPSILON, U_a, 1.0))
    return intervals.create_interval(res_L, res_U)


def bulk_and_drastic_raw(x: jnp.ndarray) -> jnp.ndarray:
    """Computes a pure drastic AND reduction along the designated terminal axis using clean optimization masks."""
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
    """Computes a pure drastic OR reduction along the designated terminal axis using clean optimization masks."""
    L = intervals.get_lower(x)
    U = intervals.get_upper(x)
    
    # Establish strict safe masks filtering sharp boundary deviations: '!= 0.0'
    not_zero_L = jnp.sum(L > EPSILON, axis=-1)
    not_zero_U = jnp.sum(U > EPSILON, axis=-1)
    
    max_L = jnp.max(L, axis=-1)
    max_U = jnp.max(U, axis=-1)
    
    # Drastic structural reduction rules for OR:
    # If all items are 0 -> output 0.0. If exactly one item is > 0 -> return that maximum item. Otherwise saturate to 1.0.
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
    max(1.0 - (1.0 - H(A)) * A, (1.0 - H(B)) * B)
    
    Evaluated independently on boundary limits conforming to strict pessimistic interval semantics.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Calculate localized structural stability weights: (1.0 - H)
    # Under pessimistic interval inversion for logical negations (NOT A):
    # The lower bound maps from U_a uncertainty, while the upper bound maps from L_a uncertainty
    w_U_a = activations.get_entropic_weight(U_a)
    w_L_a = activations.get_entropic_weight(L_a)
    w_L_b = activations.get_entropic_weight(L_b)
    w_U_b = activations.get_entropic_weight(U_b)
    
    # Synthesize the physical "NOT A" and physical "B" intermediate representations
    not_a_L = 1.0 - (w_U_a * U_a)
    not_a_U = 1.0 - (w_L_a * L_a)
    
    b_weighted_L = w_L_b * L_b
    b_weighted_U = w_U_b * U_b
    
    # Resolve the implication using the traditional Kleene-Dienes maximum primitive over deformed fields
    res_L = jnp.maximum(not_a_L, b_weighted_L)
    res_U = jnp.maximum(not_a_U, b_weighted_U)
    
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Physical Reichenbach implication via a smooth gravitational polynomial.
    
    Mathematical formula:
    1.0 - A + A * B * (1.0 - H(A) * H(B))
    
    During peak systemic uncertainty (where both signals approach the logical midpoint 0.5), 
    the coupling factor collapses entirely. This dynamic space bending establishes an uninhibited 
    linear gradient channel to facilitate highly stable error backpropagation.
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Extract structural entropy fields from boundary coordinates
    h_L_a, h_U_a = activations.entropy_raw(L_a), activations.entropy_raw(U_a)
    h_L_b, h_U_b = activations.entropy_raw(L_b), activations.entropy_raw(U_b)
    
    # Formulate localized space-bending coefficients regulating the interactive product term: (A * B)
    # The pessimistic lower bound aggregates matching lower-bound systemic uncertainties
    deform_L = 1.0 - (h_L_a * h_L_b)
    deform_U = 1.0 - (h_U_a * h_U_b)
    
    # Compute the Reichenbach polynomial: 1.0 - A + (A * B * Deform) over interval configurations
    # Lower bound (pessimistic): Minimize output -> subtract maximal U_a, add minimal product term
    res_L = 1.0 - U_a + (L_a * L_b * deform_L)
    # Upper bound (optimistic): Maximize output -> subtract minimal L_a, add maximal product term
    res_U = 1.0 - L_a + (U_a * U_b * deform_U)
    
    return intervals.create_interval(jnp.clip(res_L, 0.0, 1.0), jnp.clip(res_U, 0.0, 1.0))


def implies_physical_gravitational_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Physical Gravitational Łukasiewicz implication.
    
    Mathematical formula:
    min(1.0, 1.0 - (1.0 - H(A)) * A + (1.0 - H(B)) * B)
    
    This operator strictly shadows classical truth mappings when input signals exhibit minimal entropy, 
    while generating full structural convergence (truth = 1.0) as states enter the entropic singularity [0.5, 0.5].
    """
    L_a, U_a = intervals.get_lower(int_a), intervals.get_upper(int_a)
    L_b, U_b = intervals.get_lower(int_b), intervals.get_upper(int_b)
    
    # Extract dynamic entropic stability weights (1.0 - H) across individual boundary vectors
    w_L_a, w_U_a = activations.get_entropic_weight(L_a), activations.get_entropic_weight(U_a)
    w_L_b, w_U_b = activations.get_entropic_weight(L_b), activations.get_entropic_weight(U_b)
    
    # Evaluate algebra over intervals ensuring rigorous limit inversion for negative expressions:
    # L_res = 1 - U_weighted_a + L_weighted_b
    # U_res = 1 - L_weighted_a + U_weighted_b
    res_L = 1.0 - (w_U_a * U_a) + (w_L_b * L_b)
    res_U = 1.0 - (w_L_a * L_a) + (w_U_b * U_b)
    
    # Enforce standard Łukasiewicz axiomatic boundary limits using parallel minimum primitives
    return intervals.create_interval(jnp.minimum(1.0, res_L), jnp.minimum(1.0, res_U))
