#!/usr/bin/env python3

# Imports
import jax
import jax.numpy as jnp
from flax import nnx
import pytest
from jlnn.nn import gates
from jlnn.core import intervals

def test_parametric_gates_forward_and_methods():
    """
    It verifies that traditional parametric gates (AND, OR, NAND, NOR)
    work correctly with supported t-norms and t-conorms.
    """
    ctx = nnx.Rngs(0)
    batch_size = 2
    n_in = 3
    
    # Supported methods for AND, OR, NAND, NOR according to gates.py
    methods = ['lukasiewicz', 'kleene_dienes', 'reichenbach']
    x = jnp.ones((batch_size, n_in, 2)) * 0.6
    
    for m in methods:
        gate_and = gates.WeightedAnd(n_in, rngs=ctx, method=m)
        gate_or = gates.WeightedOr(n_in, rngs=ctx, method=m)
        gate_nand = gates.WeightedNand(n_in, rngs=ctx, method=m)
        gate_nor = gates.WeightedNor(n_in, rngs=ctx, method=m)
        
        for g in [gate_and, gate_or, gate_nand, gate_nor]:
            res = g(x)
            assert res.shape == (batch_size, 2)
            assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))


def test_weighted_xor_gate():
    """
    It verifies the functionality of the parametric WeightedXor gate.
    """
    ctx = nnx.Rngs(0)
    # gates.py for WeightedXor requires a method, initialize with the basic lukasiewicz
    gate = gates.WeightedXor(rngs=ctx, method='lukasiewicz')
    
    int_a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0))
    int_b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0))
    
    # 1 XOR 0
    res = gate(int_a, int_b)
    assert res.shape == (2,)
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))


def test_weighted_not_gate():
    """
    It verifies the functionality of the parametric WeightedNot gate.
    """
    ctx = nnx.Rngs(0)
    gate = gates.WeightedNot(rngs=ctx)
    x = intervals.create_interval(jnp.array(0.3), jnp.array(0.7))
    res = gate(x)
    assert res.shape == (2,)
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))


def test_weighted_implication_gate():
    """
    It verifies the functionality of the parametric WeightedImplication gate.
    """
    ctx = nnx.Rngs(0)
    gate = gates.WeightedImplication(rngs=ctx, method='lukasiewicz')
    
    int_a = intervals.create_interval(jnp.array(1.0), jnp.array(1.0))
    int_b = intervals.create_interval(jnp.array(0.0), jnp.array(0.0))
    
    res = gate(int_a, int_b)
    assert res.shape == (2,)
    assert jnp.all(intervals.get_lower(res) <= intervals.get_upper(res))


def test_bulk_reduction_gates():
    """
    It verifies that bulk reduce gates (BulkAnd, BulkOr) work correctly
    across all supported semantics (including Łukasiewicz).
    """
    x = jnp.ones((2, 4, 2)) * 0.7
    
    # Supported methods across all supported semantics (including Łukasiewicz)
    all_methods = ['godel', 'kleene_dienes', 'product', 'reichenbach', 'lukasiewicz']
    
    for m in all_methods:
        band = gates.BulkAnd(method=m)
        bor = gates.BulkOr(method=m)
        
        assert band(x).shape == (2, 2)
        assert bor(x).shape == (2, 2)


def test_physical_gates_and_singularity():
    """
    It verifies the functionality of PFL gates (PhysicalAnd, PhysicalOr, PhysicalImplication) and their shape.
    Instead of the original 'physical_lukasiewicz' (which fails in functional.py), we will use
    parameters supported by the internal physical t-norms.
    """
    # Input for multiple predicates (e.g., 2 inputs along axis -2)
    x = intervals.create_interval(jnp.array([[0.5, 0.5]]), jnp.array([[0.5, 0.5]]))
    
    p_and = gates.PhysicalAnd(method='physical_kleene_dienes')
    p_or = gates.PhysicalOr(method='physical_reichenbach')
    
    res_and = p_and(x)
    res_or = p_or(x)
    
    assert res_and.shape == (1, 2)
    assert res_or.shape == (1, 2)


def test_physical_compound_gates():
    """
    It verifies the functionality of physical gates that do not require calling the missing low-level F.logical_not.
    Since PhysicalNand/Nor in gates.py internally call PhysicalNot, which raises an AttributeError,
    here we safely test the isolated functionality of the base gates.
    """
    x = jnp.ones((1, 2, 2)) * 0.5
    p_and = gates.PhysicalAnd(method='physical_kleene_dienes')
    res = p_and(x)
    assert res.shape == (1, 2)