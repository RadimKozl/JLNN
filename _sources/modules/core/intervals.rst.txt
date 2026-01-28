Interval Arithmetic
===================

.. automodule:: jlnn.core.intervals
   :members:
   :undoc-members:
   :show-inheritance:

Module for working with truth intervals :math:`[L, U]`. In JLNN, truth is not just a single number, but a range that allows for representing uncertainty.

Key operations
---------------

* **Creation**: Using ``create_interval`` combines the lower and upper bounds into a single JAX tensor.
* **Extraction**: Functions ``get_lower`` and ``get_upper`` provide safe access to the bounds.
* **Negation**: Implements logical NOT as :math:`[1-U, 1-L]`, preserving the interval width (degree of ignorance).