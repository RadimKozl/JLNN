JLNN Compiler
=============

.. automodule:: jlnn.symbolic.compiler
   :members:
   :show-inheritance:

The compiler recursively traverses the syntax tree from the parser and creates corresponding **NNX modules** for each node. The result is a hierarchical model that can be trained directly.

Key Components:
------------------
* **Node**: Abstract base for all graph nodes. Ensures recursive ``forward`` calls.
* **PredicateNode**: Represents input variables (leaves of the graph).
* **NAryGateNode**: Zapouzdřuje n-ární hradla (AND, OR, XOR).

Main class that unifies the process from a string formula to an executable model.