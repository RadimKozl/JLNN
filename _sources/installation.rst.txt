==============
Installation
==============

JLNN requires Python 3.11 or later. Since the framework is built natively on the modern **JAX (0.8.2+)** stack, the installation is split into hardware-specific targets to optimize performance and prevent runtime environment conflicts.

Standard Installation
-----------------------

You can install the lightweight core library directly from PyPI:

.. code-block:: bash

    pip install jax-lnn

Alternatively, for the latest bleeding-edge version, install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/RadimKozl/JLNN.git

Hardware Acceleration (Recommended)
--------------------------------------

To leverage the full power of GPUs or TPUs without manually dealing with low-level Nvidia or Google Cloud drivers, use our built-in hardware targets:

**For NVIDIA GPU (CUDA 12+) support:**

.. code-block:: bash

    pip install "jax-lnn[gpu,export]"

**For Google Cloud TPU support:**

.. code-block:: bash

    pip install "jax-lnn[tpu,export]"

**For CPU-only machines:**

.. code-block:: bash

    pip install "jax-lnn[cpu]"

Development Installation
---------------------------

If you want to contribute to JLNN, run benchmarks, or modify the source code, clone the repository and install the development extras.

Using **uv** (recommended for blistering speed):

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd JLNN
    uv sync --all-extras

Using **pip**:

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd JLNN
    pip install -e ".[test,docs,export,extra]"

Google Colab Quickstart
--------------------------

JLNN is fully optimized for **Google Colab** (T4, V100, and premium A100 instances). To set up your environment cleanly without triggering WebSocket disconnects:

1. Create a new notebook and set your Runtime type to **GPU** or **TPU** (*Runtime -> Change runtime type*).
2. Run the following command in the first cell:

.. code-block:: bash

    !pip install "jax-lnn[gpu,export]" --quiet

3. **CRITICAL STEP:** In the top menu, click **Runtime -> Restart session** (or press ``Ctrl + M .``) to allow Python to initialize the newly installed CUDA/XLA layers.

.. warning::
   Do **not** use automated hard-kill commands like ``os.kill(os.getpid(), 9)`` on premium Colab instances (like A100). Due to high-speed virtualized network IO connections, killing the kernel process abruptly will crash the underlying WebSocket connection, throwing a terminal ``[object CloseEvent]`` error in your browser. Always restart the session gracefully via the UI menu.

Core Dependencies
--------------------

The framework automatically manages its modular ecosystem:
* **flax (>=0.12.2)**: For managing neural network states using the modern NNX API.
* **lark**: For parsing symbolic logical formulas.
* **networkx**: For handling hierarchical graph structures of formulas.
* **optax**: For neuro-symbolic parameter optimization.
* **orbax-checkpoint**: Included in ``[extra]`` for high-performance weight checkpointing (100% TensorFlow-free, eliminating C++ symbol collision issues).

Verification
---------------

To verify that your installation successfully hooked into your hardware backend, execute:

.. code-block:: python

    import jlnn
    import jax
    print(f"✅ JLNN version: {jlnn.__version__}")
    print(f"🔥 Active backend: {jax.devices()[0].platform.upper()}")
    print(f"📊 Available devices: {jax.devices()}")

OS Support
-------------

* **Linux**: Full native support (highly recommended for production training).
* **macOS**: Support for Apple Silicon chips (M1/M2/M3) via Metal shader acceleration.
* **Windows**: Supported natively via standard CPU wheels. For multi-GPU hardware training on Windows, using **WSL2 (Windows Subsystem for Linux)** is strictly recommended.