#!/usr/bin/env python3
import json
from jlnn.storage.metadata import save_metadata, load_metadata

def test_metadata_save_load_cycle(tmp_path):
    """
    Verifies that model metadata (predicate names, semantics) is preserved.
    
    Ensures that the mapping between neural indices and symbolic labels
    remains consistent after being serialized to disk.
    """
    metadata_file = tmp_path / "model_meta.json"
    sample_data = {
        "predicate_names": ["is_cat", "has_fur", "meows"],
        "logic_semantics": "lukasiewicz",
        "version": 1.2
    }

    # Save to temporary path
    save_metadata(sample_data, metadata_file)
    assert metadata_file.exists(), "Metadata file should be created on disk"

    # Load and compare
    loaded_data = load_metadata(metadata_file)
    assert loaded_data["predicate_names"] == sample_data["predicate_names"]
    assert loaded_data["logic_semantics"] == "lukasiewicz"
    assert loaded_data["version"] == 1.2

def test_load_metadata_nonexistent():
    """Ensures load_metadata raises FileNotFoundError for missing files."""
    import pytest
    with pytest.raises(FileNotFoundError):
        load_metadata("non_existent_path_999.json")