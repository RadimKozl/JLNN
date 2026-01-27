Logical Kernels
===============

.. automodule:: jlnn.core.logic
   :members:
   :undoc-members:
   :show-inheritance:

This module implements low-level mathematical definitions of logical operators. It focuses primarily on **Łukasiewicz logic**, which is the basis for LNN.

Supported semantics
-------------------

* **Łukasiewicz (Linear/Optimistic)**: Standard for JLNN. Provides stable gradients and is highly interpretable.
* **Interval reasoning**: Computations are designed to correctly propagate uncertainty through the entire network. For example, in an AND gate, the lower bound of the result is derived from the most pessimistic combination of inputs.

.. note::
   All functions in this module are designed as "pure functions" (pure functions) for maximum compatibility with ``jax.jit``.