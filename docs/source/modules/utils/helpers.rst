Helper Functions
================

.. automodule:: jlnn.utils.helpers
   :members:
   :undoc-members:

Helper functions for transforming data between standard formats and the JLNN interval representation.

* **scalar_to_interval**: Converts classical probabilities in the range :math:`[0, 1]` to precise intervals :math:`[x, x]`. This is crucial when initializing facts from fixed datasets.
* **is_precise**: Checks whether the interval has already "collapsed" into a single point (zero uncertainty).

.. autofunction:: scalar_to_interval
.. autofunction:: is_precise