Graph & NetworkX Integration
============================

.. automodule:: jlnn.symbolic.graph
   :members:
   :show-inheritance:

This module enables bidirectional conversion between the internal structure of JLNN and the graph library **NetworkX**. This is essential for model visualization and integration with existing knowledge graphs.

Export and Visualization
------------------------
The function ``build_networkx_graph`` transforms the hierarchy of NNX modules into a directed acyclic graph (DAG). Each node in the graph carries metadata about its type and label.

Import topology
----------------
Thanks to the function ``from_networkx_to_jlnn``, it is possible to create a JLNN model directly from a NetworkX graph structure, allowing the construction of logical networks without needing to write textual formulas.
