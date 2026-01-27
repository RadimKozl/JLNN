Logical Loss Functions
======================

.. automodule:: jlnn.training.losses
   :members:
   :undoc-members:
   :show-inheritance:

Loss functions in JLNN are designed to motivate the model to seek consistent interpretations of data.

Key Functions:
---------------

* **Contradiction Loss**: The most important function for stability. Penalizes states where the lower bound (L) exceeds the upper bound (U). A logical contradiction :math:`L > U` is not allowed in JLNN.
* **Uncertainty Penalization**: Motivates the model to shrink the intervals of truth (approaching L and U to each other), thereby reducing the system's "ignorance".
* **Rule Violation Loss**: Specific loss for knowledge engineering. Penalizes situations where the premise (A) is true, but the conclusion (B) is false, enforcing the validity of logical rules.

.. autofunction:: contradiction_loss
.. autofunction:: rule_violation_loss