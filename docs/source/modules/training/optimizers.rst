Projected Optimizers
====================

.. automodule:: jlnn.training.optimizers
   :members:
   :show-inheritance:

Standard optimizers (like Adam or SGD) can during training push parameters into an uninterpretable state. ``ProjectedOptimizer`` solves this problem using the technique of **Projected Gradient Descent**.

Mechanism of operation:
-----------------------

1. The standard scale update step is performed (e.g. using Optax).
2. Immediately applies projection using ``apply_constraints``.
3. Gate weights are returned to the space :math:`w \geq 1` and predicate bounds are fixed.

.. autoclass:: ProjectedOptimizer
   :members: step

   Wrapper over any ``optax`` optimizer. Ensures that the model satisfies logical axioms after each step.