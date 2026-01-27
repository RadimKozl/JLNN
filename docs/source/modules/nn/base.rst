Base Logical Elements
=====================

.. automodule:: jlnn.nn.base
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

This module defines an abstract base class for all logical components in JLNN.
It uses **Flax NNX** for object-oriented state management, which allows for native parameter handling within JAX.

Key features
------------

* **NNX Integration**: The class inherits from ``nnx.Module``, meaning weights and beta are automatically tracked as ``nnx.Param``.
* **Interval Contract**: Expects inputs in the format ``(..., n_inputs, 2)`` and returns an output interval ``[L, U]``.
* **Initialization**: Weights are initialized to the value **1.0** (neutral influence) according to the LNN standard, to prevent accidental bias in logical reasoning before training.

LogicalElement class
--------------------

.. autoclass:: jlnn.nn.base.LogicalElement
   :members:
   :noindex: