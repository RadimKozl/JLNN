Contributing to JLNN
====================

Thank you for your interest in contributing to the **JLNN** project! As a neuro-symbolic framework built on JAX, we emphasize mathematical correctness, performance, and code readability.

How can you help?
------------------

1. **Implementation of new gates**: Adding additional semantics (e.g. Gödel, Product logic) to the ``jlnn.nn.functional`` module.
2. **Temporal logic**: Extending the ``jlnn.reasoning.temporal`` module with operators such as *Until* or *Since*.
3. **Optimization**: Improving the compiler's tracer or adding support for distributed learning.
4. **Documentation**: Fixing errors in text or creating new tutorials (Jupyter Notebook).

Development Environment
-----------------------

For development, we recommend creating a clean virtual environment:

.. code-block:: bash

    git clone https://github.com/RadimKozl/JLNN.git
    cd jlnn
    python -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"

Code standards
--------------

To make the project sustainable, we adhere to the following rules:

* **Typing**: All functions must have type annotations (Type Hints) according to PEP 484.
* **Docstrings**: We use the **Google Style Python Docstrings** format. Each logical operation must have a clear mathematical explanation.
* **JAX Compliance**: Code must be purely functional where required (compatibility with ``jax.jit`` and ``jax.vmap``).
* **NNX State**: When working with parameters in modules, always use ``nnx.Param`` to ensure automatic weight tracking.

Process Pull Request
--------------------

1. Create a new branch (branch) from ``main`` with a name describing your change (e.g. ``feature/new-gate``).
2. Add tests to the ``tests/`` folder (we use ``pytest``).
3. Ensure your change does not break existing logical axioms (run checks using ``jlnn.nn.constraints``).
4. Submit a Pull Request and wait for review.

Error reporting
---------------

If you encounter numerical instability (e.g. NaN during gradient computation for XOR) or a parsing error in formulas, please open an **Issue** on GitHub with a minimal reproducible example.