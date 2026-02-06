Installation
============

JLNN requires Python 3.11 or later. Since the framework is built on the **JAX** stack, we recommend paying attention to choosing the correct version based on your hardware.

Standard Installation
---------------------

You can install the latest stable version directly from PyPI:

.. code-block:: bash

    pip install jax-lnn

Alternatively, for the latest bleeding-edge version, install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/RadimKozl/JLNN.git

Development Installation
------------------------

If you want to contribute to JLNN, run benchmarks, or modify the source code, we recommend a development installation.

Using **uv** (recommended):

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd JLNN
    uv sync

Using **pip**:

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd JLNN
    pip install -e ".[test]"

JAX & CUDA Specifics
--------------------

JLNN runs best on accelerators (GPU/TPU). In **Google Colab**, the default installation might require a runtime restart to initialize CUDA correctly.

**Example for GPU (CUDA 12) support:**

.. code-block:: bash

    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

.. note::
   If you are using Google Colab, remember to restart your session after installation to ensure JAX can access the GPU. You can do this programmatically:
   ``import os; os.kill(os.getpid(), 9)``

Dependencies
------------

The framework automatically installs these key libraries:
* **jax & jaxlib**: Computing core.
* **flax**: For managing the state of neural networks (we use the modern NNX API).
* **lark**: For parsing logical formulas.
* **networkx**: For working with the graph structure of the model.
* **optax**: For optimization and learning.

Verification
------------

To verify that everything works correctly, try a simple import:

.. code-block:: python

    import jlnn
    import jax
    print(f"JLNN version: {jlnn.__version__}")
    print(f"Available devices: {jax.devices()}")

Support for OS
---------------

* **Linux**: Full support (recommended).
* **macOS**: Support for Apple Silicon processors (M1/M2/M3) via Metal acceleration.
* **Windows**: Support via WSL2 (Windows Subsystem for Linux) is recommended for GPU acceleration.