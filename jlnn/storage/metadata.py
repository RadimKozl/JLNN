#!/usr/bin/env python3

# Imports
import json
from pathlib import Path
from typing import Any, Dict, Union

def save_metadata(metadata: Dict[str, Any], filepath: Union[str, Path]):
    """
    Serializes and saves model configuration metadata to a JSON file.

    In the context of Logical Neural Networks (LNN), metadata is essential for 
    mapping the numerical indices of the network back to their symbolic 
    meanings (e.g., mapping input index 0 to the predicate 'is_red'). 
    This function ensures that the conceptual structure of the model is 
    preserved alongside the trained weights.

    Args:
        metadata (Dict[str, Any]): A dictionary containing metadata such as:
            - 'predicate_names': List of strings labeling the input features.
            - 'logic_semantics': The t-norm type used (e.g., 'lukasiewicz').
            - 'model_version': Versioning for tracking experiments.
        filepath (Union[str, Path]): Destination path for the .json file.
            The directory tree will be created automatically if it doesn't exist.

    Example:
        >>> meta = {"predicates": ["low_temp", "high_pressure"], "version": 1.0}
        >>> save_metadata(meta, "storage/model_meta.json")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        # indent=4 ensures the file is human-readable for manual inspection
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Metadata successfully saved to: {path}")

def load_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Retrieves and deserializes metadata from a JSON file.

    This function is typically called during model reconstruction or when 
    exporting the network to symbolic rules. It allows the 'export' module 
    to label the trained weights with their corresponding real-world names.

    Args:
        filepath (Union[str, Path]): Path to the source .json file.

    Returns:
        Dict[str, Any]: The metadata dictionary containing model configuration.

    Raises:
        FileNotFoundError: If the specified metadata file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Metadata successfully loaded from: {path}")
    return data