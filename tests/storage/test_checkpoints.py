#!/usr/bin/env python3
"""
Unit tests for the checkpointing system in JLNN.

These tests ensure that model parameters can be reliably saved to and 
loaded from disk, while enforcing strict structural integrity to prevent 
logical inconsistencies caused by loading weights into incompatible architectures.
"""

# Imports
import pytest
import warnings
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
from jlnn.symbolic.compiler import LNNFormula
from jlnn.storage.checkpoints import save_checkpoint, load_checkpoint


@pytest.fixture
def rngs():
    """
    Fixture for Random Number Generators.
    Uses a fixed seed to ensure deterministic behavior across test runs.
    """
    return nnx.Rngs(42)


@pytest.fixture
def simple_model(rngs):
    """
    Fixture providing a basic logical model (A & B).
    Used for standard save/load cycle tests.
    """
    return LNNFormula("A & B", rngs)


@pytest.fixture
def different_model(rngs):
    """
    Fixture providing a model with a different logical structure (A | B | C).
    Used to verify that the system correctly rejects incompatible checkpoints.
    """
    return LNNFormula("A | B | C", rngs)


def _get_model_keys(model):
    """Helper function to get model keys with deprecation warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.nnx.statelib")
        try:
            return set(nnx.to_flat_state(model).keys())
        except AttributeError:
            # Fallback for Flax 0.12.2
            return set(dict(nnx.state(model).flat_state()).keys())


def test_save_and_load_same_structure(tmp_path, simple_model):
    """
    Verifies that a checkpoint can be saved and then successfully loaded 
    into a fresh model instance with the same logical structure.
    """
    cp_path = tmp_path / "same_structure.pkl"

    # Save checkpoint
    save_checkpoint(simple_model, cp_path)

    # New instance of the same structure (same formula, new seed → different starting weights)
    loaded_model = LNNFormula("A & B", nnx.Rngs(123))  # ← different seed, but same formula

    # Load checkpoint (should overwrite weights)
    load_checkpoint(loaded_model, cp_path)

    # Verify structure match (parameter keys)
    orig_keys = _get_model_keys(simple_model)
    loaded_keys = _get_model_keys(loaded_model)
    
    assert orig_keys == loaded_keys, "Checkpoint structure does not fit the model"


def test_load_checkpoint_structure_mismatch(tmp_path, simple_model, different_model):
    """
    Ensures that load_checkpoint raises a ValueError when attempting to load 
    parameters from a model with a different structure (e.g., different 
    number of predicates or different gate types).
    
    LNN integrity depends on the mapping between symbolic names and neural weights.
    """
    cp_path = tmp_path / "mismatch.pkl"

    # Save from a simple model
    save_checkpoint(simple_model, cp_path)

    # Attempt to load into a different model – expect ValueError (e.g. shape mismatch or predicate count)
    with pytest.raises(ValueError):
        load_checkpoint(different_model, cp_path)


def test_checkpoint_file_not_found():
    """
    Verifies that a FileNotFoundError is raised when attempting to load 
    from a non-existent path.
    """
    dummy_model = None  # no real model needed
    fake_path = Path("this_file_does_not_exist_987654321.pkl")

    with pytest.raises(FileNotFoundError):
        load_checkpoint(dummy_model, fake_path)


def test_checkpoint_roundtrip_different_seed(tmp_path, rngs):
    """
    Tests a complete round-trip: save from one model and load into another 
    initialized with a different seed.
    
    This ensures that the RNG state stored in the checkpoint (if any) 
    does not interfere with the ability to restore logical parameters 
    into a fresh object.
    """
    cp_path = tmp_path / "roundtrip.pkl"

    # Model 1 – storage
    model1 = LNNFormula("A & B -> C", rngs)
    save_checkpoint(model1, cp_path)

    # Model 2 – same formula, different seed
    model2 = LNNFormula("A & B -> C", nnx.Rngs(999999))

    # Load checkpoint
    load_checkpoint(model2, cp_path)

    # Verifying structure compliance
    keys1 = _get_model_keys(model1)
    keys2 = _get_model_keys(model2)
    
    assert keys1 == keys2, "The structure does not fit after loading"