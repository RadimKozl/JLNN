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
    Stateless weighted conjunction (AND) according to Łukasiewicz.
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_and_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)


def weighted_or(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Stateless weighted disjunction (OR) according to Łukasiewicz.
    """
    # Calls low-level implementation from kernel
    results = logic.weighted_or_lukasiewicz(x, weights, beta)
    return intervals.ensure_interval(results)


def weighted_not(x: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a weighted logical negation (NOT) with confidence scaling.
    """
    # 1. Pure negation: [L, U] -> [1-U, 1-L]
    negated = intervals.negate(x)
    
    # 2. Linear interpolation between negated result and maximum uncertainty [0, 1]
    l_neg = intervals.get_lower(negated) * weight + 0.0 * (1.0 - weight)
    u_neg = intervals.get_upper(negated) * weight + 1.0 * (1.0 - weight)
    
    # 3. Merge and enforce domain [0, 1] and consistency L <= U
    combined = jnp.stack([l_neg, u_neg], axis=-1)
    return intervals.ensure_interval(jnp.clip(combined, 0.0, 1.0))


def weighted_nand(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NAND calculated as the negation of weighted AND.
    """
    res_and = weighted_and(x, weights, beta)
    results = intervals.negate(res_and)
    return intervals.ensure_interval(results)


def weighted_nor(x: jnp.ndarray, weights: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted NOR calculated as the negation of the weighted OR.
    """
    res_or = weighted_or(x, weights, beta)
    results = intervals.negate(res_or)
    return intervals.ensure_interval(results)


# =====================================================================
# 2. GÖDEL LOGIC (Strict / Extremes / Min-Max)
# =====================================================================

def and_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure Godel conjunction (AND): min(A, B)."""
    return intervals.ensure_interval(logic.and_godel_pure(int_a, int_b))


def or_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure Godel disjunction (OR): max(A, B)."""
    return intervals.ensure_interval(logic.or_godel_pure(int_a, int_b))


def bulk_and_godel(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk Godel AND reduction across the last axis using min operations."""
    return intervals.ensure_interval(logic.bulk_and_godel_raw(x))


def bulk_or_godel(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk Godel OR reduction across the last axis using max operations."""
    return intervals.ensure_interval(logic.bulk_or_godel_raw(x))


def xor_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Pure Godel Exclusive OR (XOR).
    Defined via standard equivalence: (A AND NOT B) OR (NOT A AND B) using Min-Max logic.
    """
    not_a = intervals.negate(int_a)
    not_b = intervals.negate(int_b)
    left = logic.and_godel_pure(int_a, not_b)
    right = logic.and_godel_pure(not_a, int_b)
    return intervals.ensure_interval(logic.or_godel_pure(left, right))


def weighted_and_godel(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Applies expert reliability weights to inputs, then reduces via Godel (Min) AND."""
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_and_godel_raw(w_x))


def weighted_or_godel(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Applies expert reliability weights to inputs, then reduces via Godel (Max) OR."""
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_or_godel_raw(w_x))


# =====================================================================
# 3. PRODUCT LOGIC (Smooth / Probabilistic / Polynomial)
# =====================================================================

def and_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure algebraic product conjunction (AND): A * B."""
    return intervals.ensure_interval(logic.and_product_pure(int_a, int_b))


def or_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure algebraic product disjunction (OR): A + B - (A * B)."""
    return intervals.ensure_interval(logic.or_product_pure(int_a, int_b))


def bulk_and_product(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk product AND reduction across the last axis."""
    return intervals.ensure_interval(logic.bulk_and_product_raw(x))


def bulk_or_product(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk product OR reduction across the last axis."""
    return intervals.ensure_interval(logic.bulk_or_product_raw(x))


def xor_product(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """
    Pure Product Exclusive OR (XOR).
    Defined smoothly as: (A * (1-B)) + ((1-A) * B) - Overlap Coefficient.
    """
    not_a = intervals.negate(int_a)
    not_b = intervals.negate(int_b)
    left = logic.and_product_pure(int_a, not_b)
    right = logic.and_product_pure(not_a, int_b)
    return intervals.ensure_interval(logic.or_product_pure(left, right))


def weighted_and_product(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Applies expert reliability weights to inputs, then reduces via Product AND."""
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_and_product_raw(w_x))


def weighted_or_product(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Applies expert reliability weights to inputs, then reduces via Product OR."""
    w_x = intervals.create_interval(
        jnp.minimum(1.0, intervals.get_lower(x) * weights),
        jnp.minimum(1.0, intervals.get_upper(x) * weights)
    )
    return intervals.ensure_interval(logic.bulk_or_product_raw(w_x))


# =====================================================================
# 4. DRASTIC LOGIC (Boundary Extremes / Lower Analytical Barrier)
# =====================================================================

def and_drastic(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure drastic t-norm (AND). Collapses to 0.0 unless one argument equals 1.0."""
    return intervals.ensure_interval(logic.and_drastic_pure(int_a, int_b))


def or_drastic(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """Pure drastic t-conorm (OR). Saturates to 1.0 unless one argument equals 0.0."""
    return intervals.ensure_interval(logic.or_drastic_pure(int_a, int_b))


def bulk_and_drastic(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk drastic AND reduction across the last axis."""
    return intervals.ensure_interval(logic.bulk_and_drastic_raw(x))


def bulk_or_drastic(x: jnp.ndarray) -> jnp.ndarray:
    """Bulk drastic OR reduction across the last axis."""
    return intervals.ensure_interval(logic.bulk_or_drastic_raw(x))


# =====================================================================
# 5. ATOMIC IMPLICATION ATOMS (For Internal Dispatching)
# =====================================================================

def implication_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray, weights: jnp.ndarray, beta: float) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_lukasiewicz(int_a, int_b, weights, beta))


def implication_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_kleene_dienes(int_a, int_b))


def implication_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_reichenbach(int_a, int_b))


def implication_goguen(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_goguen(int_a, int_b))


def implication_godel(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_godel(int_a, int_b))


def implication_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_physical_kleene_dienes(int_a, int_b))


def implication_physical_reichenbach(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_physical_reichenbach(int_a, int_b))


def implication_physical_lukasiewicz(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    return intervals.ensure_interval(logic.implies_physical_gravitational_lukasiewicz(int_a, int_b))


# =====================================================================
# 6. SPACE-CURVED PHYSICAL FUZZY LOGIC (PFL) CONNECTIVES
# =====================================================================

def and_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """PFL Conjunction (AND) based on entropic space-curved warping over Min logic."""
    w_L_a = activations.get_entropic_weight(intervals.get_lower(int_a))
    w_U_a = activations.get_entropic_weight(intervals.get_upper(int_a))
    w_L_b = activations.get_entropic_weight(intervals.get_lower(int_b))
    w_U_b = activations.get_entropic_weight(intervals.get_upper(int_b))
    
    deformed_a = intervals.create_interval(w_L_a * intervals.get_lower(int_a), w_U_a * intervals.get_upper(int_a))
    deformed_b = intervals.create_interval(w_L_b * intervals.get_lower(int_b), w_U_b * intervals.get_upper(int_b))
    return intervals.ensure_interval(logic.and_godel_pure(deformed_a, deformed_b))


def or_physical_kleene_dienes(int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
    """PFL Disjunction (OR) based on entropic space-curved warping over Max logic."""
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
    Functional calculation of logical implication (A -> B).
    """
    # 1. Łukasiewicz odbočuje okamžitě (přijímá raw intervaly, váhy řeší vnitřně)
    if method == 'lukasiewicz':
        return logic.implies_lukasiewicz(int_a, int_b, weights, beta)
    
    # 2. Fyzikální metody (PFL) odbočují OKAMŽITĚ s čistými intervaly (ignorují parametrické váhy)
    if method == 'physical_kleene_dienes':
        results = logic.implies_physical_kleene_dienes(int_a, int_b)
        return intervals.ensure_interval(results)
    elif method == 'physical_reichenbach':
        results = logic.implies_physical_reichenbach(int_a, int_b)
        return intervals.ensure_interval(results)
    elif method == 'physical_lukasiewicz':
        results = logic.implies_physical_gravitational_lukasiewicz(int_a, int_b)
        return intervals.ensure_interval(results)

    # 3. Tradiční parametrické metody (kleene_dienes, reichenbach, goguen, godel)
    # Pro ně plně aplikujeme původní kontrakt předzpracování vah:
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
        raise ValueError(f"Method {method} is not supported.")
    
    return intervals.ensure_interval(results)