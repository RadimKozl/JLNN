Testing
=======

JLNN uses a comprehensive test suite to ensure the mathematical correctness of Łukasiewicz logic operations, the stability of neural modules, and the reliability of export pipelines.

Running Tests Locally
---------------------

To run the tests, you need to install the development dependencies. It is recommended to use a virtual environment.

.. code-block:: bash

   # Install JLNN with test dependencies
   pip install -e ".[test]"

   # Run the full suite using pytest
   python -m pytest tests -vv

Test Suite Structure
--------------------

The test suite is organized into several modules, mirroring the library structure:

* **Core Logic** (``tests/core/``): Verifies interval arithmetic and the fundamental Łukasiewicz t-norms/t-conorms.
* **Neural Components** (``tests/nn/``): Ensures Flax NNX modules (gates, predicates) behave correctly during forward passes and handle constraints.
* **Export Pipelines** (``tests/export/``): Validates the consistency of models when exported to:
    * **StableHLO**: Precision-matching between JAX and OpenXLA.
    * **ONNX**: Structural integrity of the generated computation graphs.
    * **PyTorch**: Numerical parity after JAX → ONNX → PyTorch conversion.
* **Symbolic Reasoning** (``tests/symbolic/``): Tests the parser and compiler for translating logical strings into neural architectures.
* **Storage & Utilities** (``tests/storage/``, ``tests/utils/``): Verifies checkpointing, metadata extraction, and Xarray integration.

Continuous Integration
----------------------

We use **GitHub Actions** to automatically run the entire test suite on every push and pull request to the ``main`` branch. 

The CI environment mirrors the production setup using:
* Ubuntu Latest
* Python 3.12
* CPU-based JAX and PyTorch runtimes

Automated Verification
----------------------

For critical operations like PyTorch conversion, JLNN includes built-in verification utilities:

.. code-block:: python

   from jlnn.export.torch_map import verify_pytorch_conversion

   # This utility (tested in tests/export/test_torch_map.py) 
   # automatically compares JAX and PyTorch outputs.
   results = verify_pytorch_conversion(jax_model, torch_model, sample_input)
   print(f"Max difference: {results['max_diff']}")