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
    Retrieves the saved parameter state and updates the existing model instance with it.

    The function performs an in-place update using `nnx.update`. 
    It is essential that the model structure (number of gates, number of inputs) matches 
    the structure stored in the checkpoint. After loading, 
    it is recommended to run `apply_constraints` to guarantee the logical integrity of the intervals.

    Args:
        model (nnx.Module): An existing model instance into which weights and biases will be loaded.
        filepath (Union[str, Path]): Path to the source .pkl file.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the state in the file is not compatible with the model architecture.

    Note:
        When loading an n-ary XOR, the function recursively restores the weights 
        at all levels of the reduction tree.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Unable to load checkpoint, file does not exist: {path}")
        
    with open(path, 'rb') as f:
        state = pickle.load(f)
    
    # nnx.update maps values ​​from State back to nnx.Param objects in the model
    nnx.update(model, state)
    print(f"Model parameters successfully restored from: {path}")