Logical Metrics
===============

.. automodule:: jlnn.utils.metrics
   :members:
   :undoc-members:

This module defines metrics for quantifying the quality of logical reasoning.

* **Contradiction Degree**: Measures the extent of violation of the axiom :math:`L \le U`. A value of 0 indicates a consistent model.
* **Uncertainty Width**: Measures the degree of "ignorance" in the model. The wider the interval :math:`U - L`, the less information the model has about a given fact.