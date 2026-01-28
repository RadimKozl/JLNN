Functional Logic Kernels
========================

.. automodule:: jlnn.nn.functional
   :members:
   :undoc-members:
   :show-inheritance:

This module provides a functional interface to all logical operations supported by the JLNN framework. Unlike the modules in :doc:`base`, these functions do not maintain their own state and are designed to be called directly with parameters passed as tensors.

Key features
------------

* **JAX-Native**: All functions are fully compatible with JAX transformations such as ``jit``, ``vmap`` and ``grad``.
* **Łukasiewicz semantics**: The default implementation of gates (AND, OR, Implication) uses Łukasiewicz logic for working with truth intervals.
* **Flexibility**: The module supports different types of implications (Kleene-Dienes, Reichenbach) for modeling expert systems.

Overview of Gates
-----------------
Negation first applies a weight to the boundaries of the interval and then performs the inversion: :math:`[1-U_w, 1-L_w]`.

Supports multiple methods for calculating the truth of the rule :math:`A \to B`. The ``lukasiewicz`` method is recommended for consistency with weights and the :math:`\beta` threshold.

Advanced operations
-------------------

Compound gates are implemented as negations of basic operators:

* **NAND**: Implemented as ``negate(weighted_and(...))``.
* **NOR**: Implemented as ``negate(weighted_or(...))``.