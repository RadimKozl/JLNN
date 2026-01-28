Metadata & State Export
=======================

.. automodule:: jlnn.export.data
   :members:
   :show-inheritance:

This module handles the serialization of model state (parameters) and its topology. It allows saving a trained model so that it can be later reconstructed without recompiling from formulas.

* **Weight Serialization**: Exporting ``nnx.Param`` (weights and betas) to Msgpack or JSON format.
* **Topology Export**: Saving the graph structure for visualization or static analysis purposes.