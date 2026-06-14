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
        jnp.ndarray: Bounded, consistency-verified truth interval [L, U] structured as (..., 2).
    """
    # Delegate implementation directly to the kernel layer
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
        jnp.ndarray: Bounded, consistency-verified truth interval [L, U] structured as (..., 2).
    """
    # Delegate implementation directly to the kernel layer
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
    # 1. Execute standard pure logical inversion: [L, U] -> [1-U, 1-L]
    negated = intervals.negate(x)
    
    # 2. Linear interpolation between the inverted result and max entropic uncertainty [0.0, 1.0]
    l_neg = intervals.get_lower(negated) * weight + 0.0 * (1.0 - weight)
    u_neg = intervals.get_upper(negated) * weight + 1.0 * (1.0 - weight)
    
    # 3. Consolidate tensor structures, enforce unit domain constraints, and verify L <= U
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
        jnp.ndarray: Bounded, consistency-verified inverted conjunction truth interval structured as (..., 2).
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
        jnp.ndarray: Bounded, consistency-verified inverted disjunction truth interval structured as (..., 2).
    """
    res_or = weighted_or(x, weights, beta)
    results = intervals.negate(res_or)
    return intervals.ensure_interval(results)


def weighted_xor_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted parametric Exclusive OR (XOR) under Łukasiewicz semantics.
    
    Formulated compositionally using standard logical equivalence axioms: 
        (A AND NOT B) OR (NOT A AND B)

    Args:
        int_a (jnp.ndarray): First input interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second input interval tensor structured as (..., 2).
        weights (jnp.ndarray): Input importance weights structured as (num_inputs,).
        beta (jnp.ndarray): Threshold parameter managing gate sensitivity.

    Returns:
        jnp.ndarray: Verified, bounded exclusive disjunction truth interval structured as (..., 2).
    """
    not_a = intervals.negate(int_a)
    not_b = intervals.negate(int_b)
    
    # Restructure input tensors to align with the (..., num_inputs, 2) contract for weighted reductions
    left_branch = jnp.stack([int_a, not_b], axis=-2)
    right_branch = jnp.stack([not_a, int_b], axis=-2)
    
    left_res = weighted_and(left_branch, weights, beta)
    right_res = weighted_and(right_branch, weights, beta)
    
    combined = jnp.stack([left_res, right_res], axis=-2)
    return weighted_or(combined, weights, beta)

def and_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure parameterless Łukasiewicz conjunction (AND) for two truth intervals.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Bounded, consistency-verified nilpotent t-norm interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.and_lukasiewicz_pure(int_a, int_b))


def or_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a pure parameterless Łukasiewicz disjunction (OR) for two truth intervals.

    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).

    Returns:
        jnp.ndarray: Bounded, consistency-verified nilpotent t-conorm interval structured as (..., 2).
    """
    return intervals.ensure_interval(logic.or_lukasiewicz_pure(int_a, int_b))


# =====================================================================
# 2. KLEENE-DIENES LOGIC CORE (Parametric Max-Min / Strict Threshold)
# =====================================================================

def weighted_and_kleene_dienes(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Parametric alias routing to the underlying weighted Gödel conjunction."""
    return weighted_and_godel(x, weights)


def weighted_or_kleene_dienes(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Parametric alias routing to the underlying weighted Gödel disjunction."""
    return weighted_or_godel(x, weights)


def weighted_nand_kleene_dienes(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Computes a weighted alternative denial (NAND) under Kleene-Dienes semantics."""
    res_and = weighted_and_kleene_dienes(x, weights)
    return intervals.ensure_interval(intervals.negate(res_and))


def weighted_nor_kleene_dienes(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Computes a weighted joint denial (NOR) under Kleene-Dienes semantics."""
    res_or = weighted_or_kleene_dienes(x, weights)
    return intervals.ensure_interval(intervals.negate(res_or))


def weighted_xor_godel(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted parametric Exclusive OR (XOR) under Gödel/Kleene-Dienes sémantics.
    
    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).
        weights (jnp.ndarray): Input reliability importance scaling weights structured as (2,).

    Returns:
        jnp.ndarray: Verified exclusive disjunction truth interval structured as (..., 2).
    """
    w_a = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_a) * weights[0]),
        jnp.minimum(1.0, intervals.get_upper(int_a) * weights[0])
    )
    w_b = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_b) * weights[1]),
        jnp.minimum(1.0, intervals.get_upper(int_b) * weights[1])
    )
    return xor_godel(w_a, w_b)


# =====================================================================
# 3. REICHENBACH LOGIC CORE (Parametric Smooth Product / Algebraic)
# =====================================================================

def weighted_and_reichenbach(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Parametric alias routing to the underlying weighted Product conjunction."""
    return weighted_and_product(x, weights)


def weighted_or_reichenbach(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Parametric alias routing to the underlying weighted Product disjunction."""
    return weighted_or_product(x, weights)


def weighted_nand_reichenbach(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Computes a weighted alternative denial (NAND) under Reichenbach semantics."""
    res_and = weighted_and_reichenbach(x, weights)
    return intervals.ensure_interval(intervals.negate(res_and))


def weighted_nor_reichenbach(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted parametric joint denial (NOR) under Reichenbach semantics.

    Args:
        x (jnp.ndarray): Input truth interval tensor structured as (..., num_inputs, 2).
        weights (jnp.ndarray): Input importance weights structured as (num_inputs,).

    Returns:
        jnp.ndarray: Consistency-verified inverted probabilistic sum interval structured as (..., 2).
    """
    res_or = weighted_or_reichenbach(x, weights)
    results = intervals.negate(res_or)
    return intervals.ensure_interval(results)


def weighted_xor_product(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted parametric Exclusive OR (XOR) under Product/Reichenbach sémantics.
    
    Args:
        int_a (jnp.ndarray): First truth interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Second truth interval tensor structured as (..., 2).
        weights (jnp.ndarray): Input reliability importance scaling weights structured as (2,).

    Returns:
        jnp.ndarray: Verified smooth exclusive disjunction truth interval structured as (..., 2).
    """
    w_a = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_a) * weights[0]),
        jnp.minimum(1.0, intervals.get_upper(int_a) * weights[0])
    )
    w_b = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(int_b) * weights[1]),
        jnp.minimum(1.0, intervals.get_upper(int_b) * weights[1])
    )
    return xor_product(w_a, w_b)


# =====================================================================
# 4. GÖDEL LOGIC (Strict / Extremes / Min-Max)
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
# 5. PRODUCT LOGIC (Smooth / Probabilistic / Polynomial)
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
# 6. DRASTIC LOGIC (Boundary Extremes / Lower Analytical Barrier)
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
# 7. ATOMIC DISJUNCTION ATOMS
# =====================================================================

def and_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Mathematical alias for and_godel (Kleene-Dienes utilizes Gödel Min t-norm)."""
    return and_godel(int_a, int_b)


def and_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Mathematical alias for and_product (Reichenbach utilizes Product t-norm)."""
    return and_product(int_a, int_b)


def or_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Mathematical alias for or_godel (Kleene-Dienes utilizes Gödel Max t-conorm)."""
    return or_godel(int_a, int_b)


def or_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Mathematical alias for or_product (Reichenbach utilizes Probabilistic Sum t-conorm)."""
    return or_product(int_a, int_b)

# =====================================================================
# 8. ATOMIC IMPLICATION ATOMS (For Internal Dispatching)
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
# 9. SPACE-CURVED PHYSICAL FUZZY LOGIC (PFL) CONNECTIVES
# =====================================================================

def and_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Conjunction (AND) based on entropic space-curved warping over Gödel Min logic.
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
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.or_godel_pure(deformed_a, deformed_b))


def and_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Conjunction (AND) based on entropic space-curved warping over Reichenbach Product logic.
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.and_product_pure(deformed_a, deformed_b))


def or_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Disjunction (OR) based on entropic space-curved warping over Reichenbach Probabilistic Sum logic.
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.or_product_pure(deformed_a, deformed_b))


def and_physical_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Conjunction (AND) based on entropic space-curved warping over Nilpotent Łukasiewicz logic.
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.and_lukasiewicz_pure(deformed_a, deformed_b))


def or_physical_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a PFL Disjunction (OR) based on entropic space-curved warping over Nilpotent Łukasiewicz logic.
    """
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.or_lukasiewicz_pure(deformed_a, deformed_b))


def logical_not(x: jnp.ndarray) -> jnp.ndarray:
    """
    Parameter-free physical negation (NOT).
    Inverts intervals securely according to: NOT [L, U] = [1 - U, 1 - L]
    """
    return intervals.ensure_interval(intervals.negate(x))


def implication(int_a: jnp.ndarray, int_b: jnp.ndarray, method: str = 'lukasiewicz') -> jnp.ndarray:
    """
    Universal proxy routing for physical or standard implication evaluations.
    Maintains compatibility with parameter-free (PFL) gates.
    """
    # Create neutral dummy weights and beta arrays since weighted_implication expects them
    dummy_weights = jnp.array([1.0, 1.0])
    dummy_beta = jnp.array(1.0)
    return weighted_implication(int_a, int_b, dummy_weights, dummy_beta, method=method)


# =====================================================================
# 10. BACKWARDS-COMPATIBLE ROUTING GATEWAY (PRESERVING ORIGINAL NAME)
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
    
    Supports traditional parametric mechanisms as well as advanced Space-Curved 
    Physical Fuzzy Logic (PFL) methods.

    Args:
        int_a (jnp.ndarray): Antecedent interval tensor structured as (..., 2).
        int_b (jnp.ndarray): Consequent interval tensor structured as (..., 2).
        weights (jnp.ndarray): Importance weights for traditional methods structured as (2,).
        beta (jnp.ndarray): Threshold/bias sensitivity parameter tensor.
        method (str): Logic framework selector string.

    Returns:
        jnp.ndarray: Bounded and consistency-verified implication result interval.
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
    # Preprocess boundaries by weighting individual operand terms according to framework contract
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
        raise ValueError(f"Implication method '{method}' is not recognized or supported.")
    
    return intervals.ensure_interval(results)