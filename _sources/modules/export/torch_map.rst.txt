PyTorch Mapping
===============

.. automodule:: jlnn.export.torch_map
   :members:

Enables mapping JLNN operations to equivalent structures in **PyTorch**. This is useful if JLNN is part of a larger system that primarily runs in the Torch ecosystem.

Key Features:
------------------
* **Weight Transfer**: Conversion of weights from JAX tensors to ``torch.nn.Parameter``.
* **Functional Mapping**: Mapping Łukasiewicz operators to Torch operations (e.g., ``torch.clamp`` instead of ``jnp.clip``).