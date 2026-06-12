#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.core import logic, intervals, activations
from typing import Optional

# =====================================================================
# 1. LUKASIEWICZ LOGIC CORE (Nilpotent / Accumulative)
# =====================================================================

def weighted_and(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Stateless weighted conjunction (AND) according to Łukasiewicz fuzzy logic.

    Acts as a functional gateway to the underlying core logic layer. It evaluates 
    the cumulative "negative evidence" across interval boundaries, where lower 
    bounds of the input restrict the upper bound of the result and vice versa.

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Input importance weights structured as (num_inputs,).
        beta (jnp.ndarray): Scalar activation sensitivity threshold parameter (bias).

    Returns:
        jnp.ndarray: Verified truth interval [L, U] structured as (..., 2).
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_and_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)


def weighted_or(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Stateless weighted disjunction (OR) according to Łukasiewicz fuzzy logic.

    Acts as a functional gateway accumulating positive validation across inputs. 
    Preserves boundary orientation where input lower bounds determine output lower bounds 
    and input upper bounds determine output upper bounds.

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Input importance weights structured as (num_inputs,).
        beta (jnp.ndarray): Scalar activation saturation threshold parameter (bias).

    Returns:
        jnp.ndarray: Verified truth interval [L, U] structured as (..., 2).
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_or_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)


def weighted_not(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted logical negation (NOT) with parameterized confidence scaling.

    First inverts the interval boundaries to simulate strict logical inversion: 
    NOT [L, U] = [1 - U, 1 - L]. Subsequently performs a linear interpolation 
    between the inverted results and the state of maximum systemic uncertainty [0.0, 1.0], 
    proportionally guided by the confidence weight factor.

    Args:
        x (jnp.ndarray): Input truth interval tensor structured as (..., 2).
        weight (jnp.ndarray): Confidence weight parameter scaling the strictness 
            of the negation mapping.

    Returns:
        jnp.ndarray: Bounded, consistency-verified truth interval structured as (..., 2).
    """
    # 1. Pure negation: [L, U] -> [1-U, 1-L]
    negated = intervals.negate(x)
    
    # 2. Linear interpolation between negated result and maximum uncertainty [0.0, 1.0]
    l_neg = intervals.get_lower(negated) * weight + 0.0 * (1.0 - weight)
    u_neg = intervals.get_upper(negated) * weight + 1.0 * (1.0 - weight)
    
    # 3. Merge and enforce domain [0.0, 1.0] and consistency L <= U
    combined = jnp.stack([l_neg, u_neg], axis=-1)
    return intervals.ensure_interval(jnp.clip(combined, 0.0, 1.0))


def weighted_nand(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted alternative denial (NAND) via the negation of a weighted Łukasiewicz AND.

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Importance weight tensor structured as (num_inputs,).
        beta (jnp.ndarray): Threshold sensitivity parameter (bias).

    Returns:
        jnp.ndarray: Consistency-verified inverted conjunction truth interval structured as (..., 2).
    """
    res_and = weighted_and(x, weights, beta)
    results = intervals.negate(res_and)
    return intervals.ensure_interval(results)


def weighted_nor(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted joint denial (NOR) via the negation of a weighted Łukasiewicz OR.

    Args:
        x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Importance weight tensor structured as (num_inputs,).
        beta (jnp.ndarray): Threshold saturation parameter (bias).

    Returns:
        jnp.ndarray: Consistency-verified inverted disjunction truth interval structured as (..., 2).
    """
    res_or = weighted_or(x, weights, beta)
    results = intervals.negate(res_or)
    return intervals.ensure_interval(results)


# =====================================================================
# 2. GÖDEL LOGIC (Strict / Extremes / Min-Max)
# =====================================================================

def and_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure Gödel conjunction (AND) for two input intervals using minimum reduction.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Verified minimum truth interval [L, U] structured as (..., 2).
    """
    return intervals.ensure_interval(logic.and_godel_pure(int_a, int_b))


def or_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure Gödel disjunction (OR) for two input intervals using maximum reduction.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Verified maximum truth interval [L, U] structured as (..., 2).
    """
    return intervals.ensure_interval(logic.or_godel_pure(int_a, int_b))


def bulk_and_godel(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk Gödel AND reduction across the designated terminal feature dimension.

    Args:
        x (jnp.ndarray): Input multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Single collapsed truth interval structured as (..., 2) containing minimum values.
    """
    return intervals.ensure_interval(logic.bulk_and_godel_raw(x))


def bulk_or_godel(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk Gödel OR reduction across the designated terminal feature dimension.

    Args:
        x (jnp.ndarray): Input multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Single collapsed truth interval structured as (..., 2) containing maximum values.
    """
    return intervals.ensure_interval(logic.bulk_or_godel_raw(x))


def xor_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure Gödel Exclusive OR (XOR) operation over truth intervals.

    Structurally formulated using the standard logical equivalence:
        (A AND NOT B) OR (NOT A AND B)
    evaluated via strict non-cumulative Min-Max fuzzy operators.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Consistency-verified exclusive disjunction truth interval structured as (..., 2).
    """
    not_a = intervals.negate(int_a)
    not_b = intervals.negate(int_b)
    left = logic.and_godel_pure(int_a, not_b)
    right = logic.and_godel_pure(not_a, int_b)
    return intervals.ensure_interval(logic.or_godel_pure(left, right))


def weighted_and_godel(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Applies expert reliability weights to inputs, then reduces them via Gödel AND.

    Scales boundaries using element-wise weights clipped at absolute truth (1.0), 
    modeling rule-based reliability adjustments before executing minimum reduction.

    Args:
        x (jnp.ndarray): Multi-variable input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Reliability scaling weights structured as (num_inputs,).

    Returns:
        jnp.ndarray: Verified truth interval representing the weighted minimum.
    """
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_and_godel_raw(w_x))


def weighted_or_godel(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Applies expert reliability weights to inputs, then reduces them via Gödel OR.

    Scales boundaries using element-wise weights clipped at absolute truth (1.0), 
    modeling rule-based reliability adjustments before executing maximum reduction.

    Args:
        x (jnp.ndarray): Multi-variable input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Reliability scaling weights structured as (num_inputs,).

    Returns:
        jnp.ndarray: Verified truth interval representing the weighted maximum.
    """
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_or_godel_raw(w_x))


# =====================================================================
# 3. PRODUCT LOGIC (Smooth / Probabilistic / Polynomial)
# =====================================================================

def and_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure algebraic product conjunction (AND) for two input intervals: A * B.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Verified product truth interval [L, U] structured as (..., 2).
    """
    return intervals.ensure_interval(logic.and_product_pure(int_a, int_b))


def or_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure algebraic product disjunction (OR): A + B - (A * B).

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Bounded and verified probabilistic sum truth interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.or_product_pure(int_a, int_b))


def bulk_and_product(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk product AND reduction across the designated terminal feature dimension.

    Args:
        x (jnp.ndarray): Input multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Single collapsed truth interval containing cumulative products.
    """
    return intervals.ensure_interval(logic.bulk_and_product_raw(x))


def bulk_or_product(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk product OR reduction across the designated terminal feature dimension.

    Args:
        x (jnp.ndarray): Input multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Single collapsed truth interval containing cumulative probabilistic sums.
    """
    return intervals.ensure_interval(logic.bulk_or_product_raw(x))


def xor_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure algebraic Product Exclusive OR (XOR) operation over truth intervals.

    Defined smoothly as: 
        (A * (1 - B)) + ((1 - A) * B) - Overlap Coefficient
    which guarantees active, non-vanishing optimization gradients throughout the unit domain.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Consistency-verified polynomial exclusive disjunction truth interval.
    """
    not_a = intervals.negate(int_a)
    not_b = intervals.negate(int_b)
    left = logic.and_product_pure(int_a, not_b)
    right = logic.and_product_pure(not_a, int_b)
    return intervals.ensure_interval(logic.or_product_pure(left, right))


def weighted_and_product(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Applies expert reliability weights to inputs, then reduces them via Product AND.

    Args:
        x (jnp.ndarray): Multi-variable input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Reliability scaling weights structured as (num_inputs,).

    Returns:
        jnp.ndarray: Verified truth interval representing the weighted algebraic product.
    """
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_and_product_raw(w_x))


def weighted_or_product(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Applies expert reliability weights to inputs, then reduces them via Product OR.

    Args:
        x (jnp.ndarray): Multi-variable input interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Reliability scaling weights structured as (num_inputs,).

    Returns:
        jnp.ndarray: Verified truth interval representing the weighted algebraic product disjunction.
    """
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_or_product_raw(w_x))


# =====================================================================
# 4. DRASTIC LOGIC (Boundary Extremes / Lower Analytical Barrier)
# =====================================================================

def and_drastic(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure drastic t-norm (AND). Collapses to 0.0 unless one argument equals exactly 1.0.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Verified drastic conjunction truth interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.and_drastic_pure(int_a, int_b))


def or_drastic(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure drastic t-conorm (OR). Saturates to 1.0 unless one argument equals exactly 0.0.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Verified drastic disjunction truth interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.or_drastic_pure(int_a, int_b))


def bulk_and_drastic(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk drastic AND reduction across the designated terminal axis.

    Args:
        x (jnp.ndarray): Multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Collapsed drastic conjunction truth interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.bulk_and_drastic_raw(x))


def bulk_or_drastic(x: jnp.ndarray) -> jnp.ndarray:
    """
    Executes a bulk drastic OR reduction across the designated terminal axis.

    Args:
        x (jnp.ndarray): Multi-variable truth interval tensor structured as (..., num_inputs, 2).

    Returns:
        jnp.ndarray: Collapsed drastic disjunction truth interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.bulk_or_drastic_raw(x))


# =====================================================================
# 5. ATOMIC IMPLICATION ATOMS (For Internal Dispatching)
# =====================================================================

def implication_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Dispatches and ensures intervals for a weighted Łukasiewicz implication."""
    return intervals.ensure_interval(logic.implies_lukasiewicz(int_a, int_b, weights, beta))


def implication_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a standard Kleene-Dienes implication."""
    return intervals.ensure_interval(logic.implies_kleene_dienes(int_a, int_b))


def implication_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a smooth Reichenbach product implication."""
    return intervals.ensure_interval(logic.implies_reichenbach(int_a, int_b))


def implication_goguen(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a residuated Goguen implication."""
    return intervals.ensure_interval(logic.implies_goguen(int_a, int_b))


def implication_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a residuated Gödel implication."""
    return intervals.ensure_interval(logic.implies_godel(int_a, int_b))


def implication_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for an entropically modulated Physical Kleene-Dienes implication."""
    return intervals.ensure_interval(logic.implies_physical_kleene_dienes(int_a, int_b))


def implication_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a gravitational Physical Reichenbach implication."""
    return intervals.ensure_interval(logic.implies_physical_reichenbach(int_a, int_b))


def implication_physical_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Dispatches and ensures intervals for a singularity-bound Physical Gravitational Łukasiewicz implication."""
    return intervals.ensure_interval(logic.implies_physical_gravitational_lukasiewicz(int_a, int_b))


# =====================================================================
# 6. SPACE-CURVED PHYSICAL FUZZY LOGIC (PFL) CONNECTIVES
# =====================================================================

def and_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Conjunction (AND) based on entropic space-curved warping over Gödel Min logic.

    Extracts individual Shannon entropy profiles from interval boundaries to formulate 
    stability mapping coefficients (1.0 - H). These coefficients deform the underlying 
    truth representations before resolving their intersection using pure minimum operations.

    Args:
        int_a (jnp.ndarray): First physical truth interval structured as (..., 2).
        int_b (jnp.ndarray): Second physical truth interval structured as (..., 2).

    Returns:
        jnp.ndarray: Bounded, consistency-verified space-warped conjunction interval structured as (..., 2).
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.and_godel_pure(deformed_a, deformed_b))


def or_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Disjunction (OR) based on entropic space-curved warping over Gödel Max logic.

    Extracts individual Shannon entropy profiles from interval boundaries to formulate 
    stability mapping coefficients (1.0 - H). These coefficients deform the underlying 
    truth representations before resolving their union using pure maximum operations.

    Args:
        int_a (jnp.ndarray): First physical truth interval structured as (..., 2).
        int_b (jnp.ndarray): Second physical truth interval structured as (..., 2).

    Returns:
        jnp.ndarray: Bounded, consistency-verified space-warped disjunction interval structured as (..., 2).
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.or_godel_pure(deformed_a, deformed_b))


# =====================================================================
# 7. BACKWARDS-COMPATIBLE ROUTING GATEWAY (PRESERVING ORIGINAL NAME)
# =====================================================================

def weighted_implication(
    int_a: jnp.ndarray, 
    int_b: jnp.ndarray, 
    weights: jnp.ndarray, 
    beta: jnp.ndarray, 
    method: str = 'lukasiewicz'
) -> jnp.ndarray:
    """
    Functional gateway for calculating structural logical implications (A -> B).

    Coordinates mathematical execution pathways between alternative fuzzy logic semantics. 
    Handles direct interval injection for accumulative systems (Łukasiewicz), parameterless 
    entropy evaluations for physical modules (PFL), and explicit boundary reliability 
    scaling for standard parametric t-norm architectures (Kleene-Dienes, Reichenbach, Goguen, Gödel).

    Args:
        int_a (jnp.ndarray): Antecedent truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Consequent truth interval tensor structured as (..., 2).
        weights (jnp.ndarray): Parameter optimization weights scaling structural rule validation. 
            Formulated as a tensor of shape (2,) representing [Weight_A, Weight_B].
        beta (jnp.ndarray): Activation threshold or sensitivity bias parameter.
        method (str): Target semantic framework selector. Options are: 'lukasiewicz', 
            'kleene_dienes', 'reichenbach', 'goguen', 'godel', 'physical_kleene_dienes', 
            'physical_reichenbach', or 'physical_lukasiewicz'. Defaults to 'lukasiewicz'.

    Returns:
        jnp.ndarray: Axiomatically guarded implication truth interval [L, U] structured as (..., 2).
        
    Raises:
        ValueError: If an unsupported semantic method string is provided.
    """
    # 1. Łukasiewicz branches immediately (accepts raw intervals, solves weights internally)
    if method == 'lukasiewicz':
        return logic.implies_lukasiewicz(int_a, int_b, weights, beta)
    
    # 2. Physical methods (PFL) branch immediately with clean intervals (ignore parametric weights)
    if method == 'physical_kleene_dienes':
        results = logic.implies_physical_kleene_dienes(int_a, int_b)
        return intervals.ensure_interval(results)
    elif method == 'physical_reichenbach':
        results = logic.implies_physical_reichenbach(int_a, int_b)
        return intervals.ensure_interval(results)
    elif method == 'physical_lukasiewicz':
        results = logic.implies_physical_gravitational_lukasiewicz(int_a, int_b)
        return intervals.ensure_interval(results)

    # 3. Traditional parametric methods (kleene_dienes, reichenbach, goguen, godel)
    # For them, we fully apply the original contract for preprocessing weights:
    weighted_a = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_a) * weights[0]),
        jnp.minimum(1.0, intervals.get_upper(int_a) * weights[0])
    )
    weighted_b = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_b) * weights[1]),
        jnp.minimum(1.0, intervals.get_upper(int_b) * weights[1])
    )

    if method == 'kleene_dienes':
        results = logic.implies_kleene_dienes(weighted_a, weighted_b)
    elif method == 'reichenbach':
        results = logic.implies_reichenbach(weighted_a, weighted_b)
    elif method == 'goguen':
        results = logic.implies_goguen(weighted_a, weighted_b)
    elif method == 'godel':
        results = logic.implies_godel(weighted_a, weighted_b)
    else:
        raise ValueError(f"Method '{method}' is not supported inside JLNN functional gateways.")
    
    return intervals.ensure_interval(results)