#!/usr/bin/env python3

# Imports
import pickle
from pathlib import Path
from flax import nnx
from typing import Union, Optional

def save_checkpoint(model: nnx.Module, filepath: Union[str, Path]):
    """
    Serializes and saves the current state of the NNX model parameters to a binary file.

    This function uses the `nnx.split` mechanism, 
    which separates the graph definition from the data itself (weights and beta parameters). 
    Only the state is stored, ensuring that the files are compact 
    and contain all the learned logic rules defined in gates like WeightedAnd or WeightedXor.

    Args:
        model (nnx.Module): An instance of the logical model whose parameters 
            (including the weights w >= 1.0 enforced in constraints.py) are to be stored.
        filepath (Union[str, Path]): The path to the target file (usually with a .pkl extension). 
            If the directory does not exist, it will be created automatically.

    Example:
        >>> model = WeightedXor(num_inputs=4, rngs=nnx.Rngs(42))
        >>> save_checkpoint(model, "checkpoints/xor_v1.pkl")
    """
    # NNX split splits the object into GraphDef and State (containing nnx.Param)
    _, state = nnx.split(model)
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        # We use pickle to preserve the structure of JAX fields within NNX State
        pickle.dump(state, f)
    print(f"Checkpoint saved successfully: {path}")

def load_checkpoint(model: nnx.Module, filepath: Union[str, Path]):
    """
    Retrieves the saved parameter state and updates the existing model instance.
    Includes structural integrity check via parameter keys and shapes.
    
    Args:
        model (nnx.Module): The model instance to load parameters into.
        filepath (Union[str, Path]): Path to the checkpoint file.
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        ValueError: If the checkpoint structure doesn't match the model structure.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Unable to load checkpoint, file does not exist: {path}")
        
    with open(path, 'rb') as f:
        loaded_state = pickle.load(f)
    
    # Use nnx.state to get the state and convert to dict for comparison
    # Note: The deprecation warning suggests using dict in the future
    current_state = nnx.state(model)
    
    # Convert FlatState to regular dict for comparison
    current_flat_dict = dict(current_state.flat_state())
    loaded_flat_dict = dict(loaded_state.flat_state())

    # 1. Kontrola shody všech klíčů (paths k parametrům)
    current_keys = set(current_flat_dict.keys())
    loaded_keys = set(loaded_flat_dict.keys())

    if current_keys != loaded_keys:
        missing = current_keys - loaded_keys
        extra = loaded_keys - current_keys
        raise ValueError(
            f"Checkpoint structure mismatch (parameter paths differ).\n"
            f"Missing in checkpoint: {missing}\n"
            f"Extra in checkpoint: {extra}"
        )

    # 2. Kontrola shapes u všech Param objektů
    for key in current_keys:
        curr_param = current_flat_dict[key]
        load_param = loaded_flat_dict[key]
        
        # Check if both have the same variable state type
        if type(curr_param) != type(load_param):
            raise ValueError(
                f"Type mismatch for parameter '{key}': "
                f"current type {type(curr_param).__name__}, "
                f"checkpoint type {type(load_param).__name__}"
            )
        
        # Check shapes for parameters with values
        # Use get_value() instead of .value to avoid deprecation warning
        try:
            curr_val = curr_param.get_value() if hasattr(curr_param, 'get_value') else curr_param[...]
            load_val = load_param.get_value() if hasattr(load_param, 'get_value') else load_param[...]
            
            curr_shape = curr_val.shape
            load_shape = load_val.shape
            
            if curr_shape != load_shape:
                raise ValueError(
                    f"Checkpoint structure mismatch: shape mismatch for parameter '{key}' "
                    f"(current: {curr_shape}, checkpoint: {load_shape})"
                )
        except (AttributeError, TypeError):
            # If the parameter doesn't have a value/shape, skip shape checking
            pass

    # Pokud vše sedí → aktualizujeme model
    nnx.update(model, loaded_state)
    print(f"Model parameters successfully restored from: {path}")