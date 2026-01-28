JAX Execution Engine
====================

.. automodule:: jlnn.reasoning.engine
   :members:
   :show-inheritance:

The ``JLNNEngine`` class serves as the main orchestrator between JAX and the logical model. It encapsulates low-level operations so that end users do not have to manage states in NNX.

Key roles of the engine:
------------------------

* **JIT Compilation**: The ``infer`` method uses ``@nnx.jit`` to transform recursive logical calls into highly optimized code for GPU/TPU.
* **Atomic Training**: The ``train_step`` method ensures that weight updates and subsequent logical projections (constraints) occur as a single indivisible operation.
* **Abstraction**: A smooth interface for passing data in dictionary form (Dict) directly into symbolic predicates.
