#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
import pytest
from jlnn.core import intervals, logic

def test_weighted_and_lukasiewicz():
    """
    Validates the weighted Łukasiewicz conjunction with multiple inputs.

    This test verifies that:
    1. When all inputs are fully True ([1.0, 1.0]) and weights are neutral (1.0), 
       the resulting interval is correctly identified as fully True.
    2. The logic properly handles weighted resistance aggregation against the beta threshold.
    """
    inputs = intervals.create_interval(jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]))
    weights = jnp.array([1.0, 1.0])
    beta = jnp.array(1.0)
    
    res = logic.weighted_and_lukasiewicz(inputs, weights, beta)
    assert jnp.allclose(intervals.get_lower(res), 1.0)
    assert jnp.allclose(intervals.get_upper(res), 1.0)


def test_lukasiewicz_pure_operators():
    """Ověří čisté bezparametrické t-normy a t-conormy Łukasiewicze."""
    a = intervals.create_interval(jnp.array(0.6), jnp.array(0.8))
    b = intervals.create_interval(jnp.array(0.5), jnp.array(0.7))
    
    # AND: max(0, A + B - 1) -> L: max(0, 0.6+0.5-1)=0.1, U: max(0, 0.8+0.7-1)=0.5
    res_and = logic.and_lukasiewicz_pure(a, b)
    assert jnp.allclose(intervals.get_lower(res_and), 0.1)
    assert jnp.allclose(intervals.get_upper(res_and), 0.5)
    
    # OR: min(1, A + B) -> L: min(1, 0.6+0.5)=1.0, U: min(1, 0.8+0.7)=1.0
    res_or = logic.or_lukasiewicz_pure(a, b)
    assert jnp.allclose(intervals.get_lower(res_or), 1.0)
    assert jnp.allclose(intervals.get_upper(res_or), 1.0)


def test_implies_kleene_dienes():
    """
    Tests the Kleene-Dienes (pessimistic) implication: max(1 - A, B).

    In interval logic, this test confirms the boundary condition for 
    classical logic within a fuzzy framework: if the antecedent is True 
    and the consequent is False, the implication must be fully False.
    """
    a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0)) # True
    b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0)) # False
    
    res = logic.implies_kleene_dienes(a, b)
    assert jnp.allclose(intervals.get_upper(res), 0.0)


def test_implies_reichenbach():
    """Verifikuje Reichenbachovu implikaci: 1 - A + (A * B)."""
    a = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    b = intervals.create_interval(jnp.array(0.4), jnp.array(0.4))
    
    # 1 - 0.5 + (0.5 * 0.4) = 0.5 + 0.2 = 0.7
    res = logic.implies_reichenbach(a, b)
    assert jnp.allclose(intervals.get_lower(res), 0.7)
    assert jnp.allclose(intervals.get_upper(res), 0.7)


def test_implies_goguen_and_godel():
    """Ověří chování Goguenovy (podílové) a Gödelovy (opoziční) implikace."""
    a = intervals.create_interval(jnp.array(0.8), jnp.array(0.8))
    b = intervals.create_interval(jnp.array(0.4), jnp.array(0.4))
    
    # Goguen: b / a pro a > b -> 0.4 / 0.8 = 0.5
    res_goguen = logic.implies_goguen(a, b)
    assert jnp.allclose(intervals.get_lower(res_goguen), 0.5)
    
    # Gödel: b pro a > b -> 0.4
    res_godel = logic.implies_godel(a, b)
    assert jnp.allclose(intervals.get_lower(res_godel), 0.4)
    
    # If a <= b, both must return 1.0
    equal_a = intervals.create_interval(jnp.array(0.3), jnp.array(0.3))
    equal_b = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    assert jnp.allclose(intervals.get_lower(logic.implies_goguen(equal_a, equal_b)), 1.0)
    assert jnp.allclose(intervals.get_lower(logic.implies_godel(equal_a, equal_b)), 1.0)


# =====================================================================
# PHYSICAL FUZZY LOGIC (PFL) OPERATOR TESTS
# =====================================================================

def test_physical_kleene_dienes_singularity():
    """Verifies the behavior of the Physical Kleene-Dienes implication at the entropic singularity point."""
    # In classical logic, 0.5 -> 0.5 with Kleene-Dienes gives max(1-0.5, 0.5) = 0.5
    # The physical PFL version must collapse to 1.0 due to 100% entropic deformation
    singularity = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    res = logic.implies_physical_kleene_dienes(singularity, singularity)
    
    assert jnp.allclose(intervals.get_lower(res), 1.0)
    assert jnp.allclose(intervals.get_upper(res), 1.0)


def test_physical_reichenbach_edges():
    """Verifies that the Physical Reichenbach correctly shadows classical behavior at the edges with zero entropy."""
    # Deterministic true antecedent and false consequent -> 1.0 -> 0.0 must remain 0.0
    a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0))
    b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0))
    
    res_edge = logic.implies_physical_reichenbach(a, b)
    assert jnp.allclose(intervals.get_lower(res_edge), 0.0)
    assert jnp.allclose(intervals.get_upper(res_edge), 0.0)
    
    # Singularity check (0.5, 0.5) -> must smoothly generate 0.5 based on product field dynamics
    singularity = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    res_sing = logic.implies_physical_reichenbach(singularity, singularity)
    assert jnp.allclose(intervals.get_lower(res_sing), 0.5)
    assert jnp.allclose(intervals.get_upper(res_sing), 0.5)


def test_implies_physical_gravitational_lukasiewicz():
    """Tests the gravitational PFL implication built on Łukasiewicz algebra."""
    # Pure edges: 1.0 -> 0.0 -> stability (1-H) is 1.0, behaves like classical Łukasiewicz: 1 - 1 + 0 = 0.0
    a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0))
    b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0))
    
    res_edge = logic.implies_physical_gravitational_lukasiewicz(a, b)
    assert jnp.allclose(intervals.get_lower(res_edge), 0.0)
    
    # Singularity: 0.5 -> 0.5 -> entropy is maximum, stability is 0.0 -> formula gives 1.0 - 0*0.5 + 0*0.5 = 1.0
    singularity = intervals.create_interval(jnp.array(0.5), jnp.array(0.5))
    res_sing = logic.implies_physical_gravitational_lukasiewicz(singularity, singularity)
    assert jnp.allclose(intervals.get_lower(res_sing), 1.0)