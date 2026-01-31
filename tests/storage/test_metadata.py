#!/usr/bin/env python3

"""
Unit tests for the metadata storage system in JLNN.

Metadata is crucial for neuro-symbolic models as it maintains the mapping 
between internal neural tensor indices and human-readable symbolic labels 
(predicates, logic rules, etc.). These tests ensure that this mapping 
is correctly serialized and deserialized.
"""

# Imports
import pytest
import json
from jlnn.storage.metadata import save_metadata, load_metadata

def test_metadata_save_load_cycle(tmp_path):
    """
    Verifies that model metadata (predicate names, semantics) is correctly 
    preserved through a full save/load cycle.
    
    This ensures that the symbolic-to-neural mapping remains consistent 
    after being stored on disk in JSON format.
    """
    metadata_file = tmp_path / "model_meta.json"
    sample_data = {
        "predicate_names": ["is_cat", "has_fur", "meows"],
        "logic_semantics": "lukasiewicz",
        "version": 1.2
    }

    # Save metadata to the temporary test directory
    save_metadata(sample_data, metadata_file)
    assert metadata_file.exists(), "Metadata file should be created on disk"

    # Load the data back and compare with the original
    loaded_data = load_metadata(metadata_file)
    assert loaded_data["predicate_names"] == sample_data["predicate_names"]
    assert loaded_data["logic_semantics"] == "lukasiewicz"
    assert loaded_data["version"] == 1.2

def test_load_metadata_nonexistent():
    """
    Ensures that load_metadata correctly raises a FileNotFoundError 
    when attempting to access a file that does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_metadata("non_existent_path_999.json")
        
def test_metadata_formatting(tmp_path):
    """
    Verifies that the saved JSON is human-readable (pretty-printed).
    
    Pretty-printing is important for manual inspection and debugging of 
    the model's symbolic configuration.
    """
    metadata_file = tmp_path / "format_test.json"
    sample_data = {"key": "value"}
    
    save_metadata(sample_data, metadata_file)
    
    # Check if the file contains newlines/indentation (standard for indent=4)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "\n" in content, "Metadata JSON should be formatted with newlines for readability."