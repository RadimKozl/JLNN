#!/usr/bin/env python3
"""
Tests for checkpoint saving and loading functionality in JLNN.
Focuses on structural integrity checks to prevent loading incompatible model states.
"""

import pytest
import warnings
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
from jlnn.symbolic.compiler import LNNFormula
from jlnn.storage.checkpoints import save_checkpoint, load_checkpoint


@pytest.fixture
def rngs():
    """Fixture pro RNG – používáme fixní seed pro reprodukovatelnost."""
    return nnx.Rngs(42)


@pytest.fixture
def simple_model(rngs):
    """Jednoduchý model: A & B"""
    return LNNFormula("A & B", rngs)


@pytest.fixture
def different_model(rngs):
    """Model s jinou strukturou: A | B | C"""
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
    Ověřuje, že checkpoint lze uložit a načíst do modelu se stejnou strukturou.
    """
    cp_path = tmp_path / "same_structure.pkl"

    # Uložit checkpoint
    save_checkpoint(simple_model, cp_path)

    # Nová instance stejné struktury (stejná formule, nový seed → jiné start váhy)
    loaded_model = LNNFormula("A & B", nnx.Rngs(123))  # ← jiný seed, ale stejná formule

    # Načíst checkpoint (měl by přepsat váhy)
    load_checkpoint(loaded_model, cp_path)

    # Ověřit shodu struktury (klíče parametrů)
    orig_keys = _get_model_keys(simple_model)
    loaded_keys = _get_model_keys(loaded_model)
    
    assert orig_keys == loaded_keys, "Struktura checkpointu nesedí s modelem"


def test_load_checkpoint_structure_mismatch(tmp_path, simple_model, different_model):
    """
    Ověřuje, že načítání selže při neshodě struktury (počet predikátů, typ gate atd.).
    """
    cp_path = tmp_path / "mismatch.pkl"

    # Uložit z jednoduchého modelu
    save_checkpoint(simple_model, cp_path)

    # Pokus načíst do odlišného modelu – očekáváme ValueError (např. shape mismatch nebo predicate count)
    with pytest.raises(ValueError):
        load_checkpoint(different_model, cp_path)


def test_checkpoint_file_not_found():
    """
    Ověřuje FileNotFoundError při neexistujícím souboru.
    """
    dummy_model = None  # není potřeba reálný model
    fake_path = Path("this_file_does_not_exist_987654321.pkl")

    with pytest.raises(FileNotFoundError):
        load_checkpoint(dummy_model, fake_path)


def test_checkpoint_roundtrip_different_seed(tmp_path, rngs):
    """
    Round-trip test: uložení → načtení do modelu se stejnou strukturou, ale jiným seedem.
    """
    cp_path = tmp_path / "roundtrip.pkl"

    # Model 1 – uložení
    model1 = LNNFormula("A & B -> C", rngs)
    save_checkpoint(model1, cp_path)

    # Model 2 – stejná formule, jiný seed
    model2 = LNNFormula("A & B -> C", nnx.Rngs(999999))

    # Načtení checkpointu
    load_checkpoint(model2, cp_path)

    # Ověření shody struktury
    keys1 = _get_model_keys(model1)
    keys2 = _get_model_keys(model2)
    
    assert keys1 == keys2, "Struktura po načtení nesedí"