Installation
============

LNN requires Python 3.9 or later. Since the framework is built on the **JAX** library, we recommend paying attention to choosing the correct version based on your hardware.

Quick installation
------------------

Quick installation can be done directly from the repository using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/RadimKozl/JLNN.git

Dependencies
------------

The framework automatically installs these key libraries:
* **jax & jaxlib**: Computing core.
* **flax**: For managing the state of neural networks (we use the modern NNX API).
* **lark**: For parsing logical formulas.
* **networkx**: For working with the graph structure of the model.
* **optax**: For optimization and learning.

JAX installation specifics
--------------------------

JLNN runs best on accelerators (GPU/TPU). If you plan to train large logical graphs, follow the official `JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`_.

**Example for GPU (CUDA 12):**

.. code-block:: bash

    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Development installation
------------------------

If you want to contribute to JLNN or run tests, install the framework in editable mode including development tools:

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd jlnn
    pip install -e ".[dev]"

Verification of installation
-----------------

To verify that everything works correctly, you can perform a simple import in the Python console:

.. code-block:: python

    import jlnn
    import jax
    print(f"JLNN verze: {jlnn.__version__}")
    print(f"Dostupná zařízení: {jax.devices()}")

Support for OS
----------

* **Linux**: Full support (recommended).
* **macOS**: Support for Apple Silicon processors (M1/M2/M3) via Metal acceleration.
* **Windows**: Support via WSL2 (Windows Subsystem for Linux) is recommended for GPU acceleration.