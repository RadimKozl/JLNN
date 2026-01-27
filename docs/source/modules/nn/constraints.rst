Parameter Constraints
=====================

.. automodule:: jlnn.nn.constraints
   :members:
   :show-inheritance:

This module implements mechanisms to ensure the logical integrity of the model during training. It uses the **Projected Gradient Descent** method, which returns the parameters to the allowed space after each step of the optimizer.

Why are restrictions necessary?
--------------------------------

Within **Logical Neural Networks (LNN)**, axiomatic conditions must be met for the model to remain interpretable as a set of logical rules:

1. **Gate weights (:math:`w \geq 1`)**:
   If a weight drops below 1.0, the gate would lose its identity (e.g., AND would stop behaving like a t-norm). The function ``clip_weights`` ensures this for all gate types.

2. **Consistency of predicates (:math:`L \leq U`)**:
   In learned predicates (LearnedPredicates), the lower bound of truth must always cover the upper bound. Mathematically, this requires the condition ``offset_u <= offset_l``. The function ``clip_predicates`` ensures that these bounds never cross.

Main Functions
--------------

.. autofunction:: apply_constraints

   This function should be called in each step of the training loop immediately after updating the weights:

   .. code-block:: python

      optimizer.update(grads)
      apply_constraints(model)  # Ensuring logical integrity

.. autofunction:: clip_weights
.. autofunction:: clip_predicates