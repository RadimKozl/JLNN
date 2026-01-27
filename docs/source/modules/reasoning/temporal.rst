Temporal Logic Operators
========================

.. automodule:: jlnn.reasoning.temporal
   :members:
   :show-inheritance:

This module implements operators inspired by Linear Temporal Logic (LTL), optimized for processing time series in JAX.

Implemented Operators:
----------------------

* **AlwaysNode (Globally)**: Implemented as a generalized conjunction (AND) over the time axis. Requires the formula to hold at all steps in the window.
* **EventuallyNode (Finally)**: Implemented as a generalized disjunction (OR). It suffices for the formula to hold at least once in a moment.

Use:
--------
Ideal for analyzing sensor data, where we seek persistent states (Always) or event detection (Eventually) in a sliding window.

.. autoclass:: AlwaysNode
   :members: forward
.. autoclass:: EventuallyNode
   :members: forward