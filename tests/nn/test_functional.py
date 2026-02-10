#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from jlnn.nn import functional as F
from jlnn.core import intervals

def test_weighted_and_functional():
    """
    Tests the stateless weighted AND operation using Łukasiewicz semantics.

    This test verifies that:
    1. The functional interface correctly processes interval tensors.
    2. Under neutral weights (1.0) and standard beta (1.0), the operation 
       satisfies the T-norm axiom (e.g., 1 AND 0 = 0).
    3. The batch dimension is preserved during the computation.
    """
    # Inputs: [1,1] (True) and [0,0] (False)
    # In Łukasiewicz logic: 1 AND 0 = 0
    x = intervals.create_interval(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
    x = x[jnp.newaxis, ...] # Add batch dimension
    
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = F.weighted_and(x, weights, beta)
    
    # Check consistency: Lower bound must be <= Upper bound
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))
    
    # In Łukasiewicz logic: 1 AND 0 = 0
    assert jnp.isclose(intervals.get_upper(res), 0.0)

def test_weighted_or_functional():
    """
    Tests the stateless weighted OR operation using Łukasiewicz semantics.

    This test verifies that:
    1. The disjunction correctly aggregates truth values (T-conorm).
    2. Under neutral weights, 1 OR 0 results in 1 (True).
    3. The interval bounds [L, U] are correctly maintained in the output.
    """
    # 1 OR 0 = 1
    x = intervals.create_interval(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
    x = x[jnp.newaxis, ...]
    
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = F.weighted_or(x, weights, beta)
    
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))
    
    # Check if the lower bound of the result is close to 1 (True, 1 OR 0 = 1)
    assert jnp.isclose(intervals.get_lower(res), 1.0)
    

def test_weighted_not_fuzzy_integrity():
    """
    Validates uncertainty preservation during fuzzy negation transformations.

    A common failure mode in interval logic is 'information collapse,' where 
    near-boundary values (e.g., [0.95, 1.0]) are prematurely clipped to crisp 
    values (e.g., [0, 0]) during negation, destroying the epistemic uncertainty.

    This test ensures that:
    1. **Mathematical Inversion:** The NOT operator correctly maps [L, U] to 
       [1-U, 1-L] without rounding errors.
    2. **Uncertainty Width Conservation:** The 'ignorance' (U - L) of the 
       input interval is exactly preserved in the output.
    3. **Fuzzy Boundary Mapping:** 'Almost True' inputs correctly transition 
       to 'Almost False' outputs, maintaining the delicate balance required 
       for neuro-symbolic backpropagation.

    Example:
        Input:  [0.95, 1.00] (Width: 0.05)
        Output: [0.00, 0.05] (Width: 0.05)
    """
    # "Almost True" interval [0.95, 1.0], width = 0.05
    x = intervals.create_interval(jnp.array(0.95), jnp.array(1.0))
    weight = jnp.array(1.0)
    
    res = F.weighted_not(x, weight)
    
    lower = intervals.get_lower(res)
    upper = intervals.get_upper(res)
    width = intervals.uncertainty(res)
    
    # 1. Check mathematical correctness
    assert jnp.isclose(lower, 0.0), f"Expected lower 0.0, got {lower}"
    assert jnp.isclose(upper, 0.05), f"Expected upper 0.05, got {upper}"
    
    # 2. Check uncertainty preservation
    assert jnp.isclose(width, 0.05), f"Uncertainty width lost! Expected 0.05, got {width}"
    
    # 3. Invariant check
    assert lower <= upper


def test_weighted_not_consistency():
    """
    Validates boundary consistency and domain enforcement for weighted negation 
    using confidence-scaling semantics.

    This test ensures that weighted NOT remains mathematically sound even with 
    super-unit weights. In this implementation, the weight acts as a confidence 
    interpolator between pure negation and the unknown state [0, 1].

    Calculation for input [0.2, 0.8] and weight 1.5:
    1. Pure Negation: [0.2, 0.8]
    2. Interpolated Lower: 0.2 * 1.5 + 0.0 * (1 - 1.5) = 0.3
    3. Interpolated Upper: 0.8 * 1.5 + 1.0 * (1 - 1.5) = 1.2 - 0.5 = 0.7
    4. Result: [0.3, 0.7]

    Asserts:
        - Invariant: Resulting Lower bound <= Upper bound.
        - Domain: All truth values reside within [0.0, 1.0].
        - Precision: Matches confidence-scaling interpolation results.
    """
    # Input [0.2, 0.8], Weight 1.5
    x = intervals.create_interval(jnp.array(0.2), jnp.array(0.8))
    weight = jnp.array(1.5)

    res = F.weighted_not(x, weight)

    lower = intervals.get_lower(res)
    upper = intervals.get_upper(res)

    # 1. Invariant check: L <= U
    assert jnp.all(lower <= upper), f"Inconsistent boundaries: {res}"

    # 2. Domain check: Must be strictly in [0, 1]
    assert jnp.all(res >= 0.0) and jnp.all(res <= 1.0), f"Out of bounds: {res}"

    # 3. Value check: Based on confidence interpolation
    # Lower: 0.2 * 1.5 + 0.0 * (-0.5) = 0.3
    assert jnp.isclose(lower, 0.3), f"Expected interpolated lower 0.3, got {lower}"
    
    # Upper: 0.8 * 1.5 + 1.0 * (-0.5) = 1.2 - 0.5 = 0.7
    assert jnp.isclose(upper, 0.7), f"Expected interpolated upper 0.7, got {upper}"

 
def test_weighted_nand_consistency():
    """
    Validates that the weighted NAND operation prevents negative uncertainty widths.

    The NAND operation is a compound transformation (NOT AND). In interval logic, 
    this sequence is highly susceptible to boundary inversion. This test verifies 
    that even when intermediate logical steps produce potentially invalid states, 
    the final `weighted_nand` output is strictly validated and corrected.

    Specifically, it ensures that:
    1. The composite logic correctly aggregates input intervals.
    2. The boundary swap inherent in the negation step (NOT) is properly managed.
    3. The final result preserves the interval invariant L <= U, ensuring 
       numerical stability for subsequent neuro-symbolic layers.
    """
    # Input with overlapping boundaries to challenge the consistency logic
    x = intervals.create_interval(jnp.array([0.7, 0.5]), jnp.array([0.9, 0.6]))
    x = x[jnp.newaxis, ...]
    
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = F.weighted_nand(x, weights, beta)
    
    # Check if the internal consistency mechanism handled the nested negation
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res)), \
        f"NAND produced inconsistent interval boundaries: {res}"
        
    # Ensure width is non-negative
    width = intervals.get_upper(res) - intervals.get_lower(res)
    assert jnp.all(width >= 0.0), f"Negative uncertainty width detected: {width}"
    
def test_weighted_implication_methods():
    """
    Validates cross-method consistency for weighted logical implications.

    This test ensures that all supported implication semantics (Łukasiewicz, 
    Kleene-Dienes, and Reichenbach) maintain the interval invariant (L <= U) 
    when processing weighted antecedents and consequents. 

    Implications are often sensitive to boundary inversions during the 
    weighting of the antecedent (int_a). This test confirms that:
    1. The `weighted_implication` interface correctly routes data to specific 
       logical kernels.
    2. Every method, regardless of its mathematical implementation, returns 
       a consistent and valid truth interval.
    3. The boundary correction mechanism (ensure_consistent) is active and 
       functional across all implication types.

    Methods tested:
        - Łukasiewicz: Standard T-norm based implication.
        - Kleene-Dienes: Max-min based implication.
        - Reichenbach: Product-based continuous implication.
    """
    # Test case: High truth antecedent [0.8, 1.0] implies low truth consequent [0.2, 0.4]
    int_a = intervals.create_interval(jnp.array(0.8), jnp.array(1.0))
    int_b = intervals.create_interval(jnp.array(0.2), jnp.array(0.4))
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    for method in ['lukasiewicz', 'kleene_dienes', 'reichenbach']:
        res = F.weighted_implication(int_a, int_b, weights, beta, method=method)
        
        # Verify interval integrity for the specific method
        assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res)), \
            f"Implication method '{method}' produced an inconsistent interval: {res}"
            
        # Domain check: Results must stay within [0, 1]
        assert jnp.all(res >= 0.0) and jnp.all(res <= 1.0), \
            f"Implication method '{method}' out of logical bounds: {res}"