Model Checkpoints
=================

.. automodule:: jlnn.storage.checkpoints
   :members:
   :undoc-members:
   :show-inheritance:

This module uses the ``nnx.split`` mechanism to separate the graph structure from the actual data (weights). This allows only essential parameters to be saved, saving space and increasing stability.

Key Features:
-------------

* **Compact storage**: Only ``nnx.Param`` type objects are stored (weights $w \ge 1$ and bias $\beta$).
* **State integrity**: After loading a checkpoint, it is recommended to run :doc:`../nn/constraints`, to ensure logical consistency even after manual file modifications.

.. note::
   When loading, the target model's structure (number of gates and inputs) must exactly match the saved state.