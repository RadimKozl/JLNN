Logical Gates (Stateful)
========================

.. automodule:: jlnn.nn.gates
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

This module contains an implementation of logic gates like **Flax NNX** modules. Each gate manages its own trainable parameters (weights and thresholds) and provides forward computation over truth intervals.

Stateful vs. Functional Gates
------------------------------

Unlike :doc:`functional`, the gates in this module:
* They store their internal state using ``nnx.Param``.
* They are compatible with automatic parameter search and optimizers.
* They are the basic building blocks that the compiler assembles into deeper structures.

Basic gates
-----------
Implements logical conjunction. Weights allow the network to selectively suppress unimportant inputs.

Implements a logical disjunction. The ``beta`` parameter determines the sensitivity (steepness) of the gate.

Weighted negation. The weight allows the model to learn how strongly to invert a given input.

Advanced operators
------------------
Allows modeling of causal relationships :math:`A \to B`. Supports various semantics (Łukasiewicz, Kleene-Dienes, Reichenbach).

Implementation of n-ary XOR using a hierarchical tree of binary operations. This structure allows learning complex parity functions.

Negated gates (NAND, NOR)
-------------------------
These gates are key for detecting logical contradictions and enforcing integrity constraints in knowledge bases.

* **WeightedNand**: Negation of conjunction.
* **WeightedNor**: Negation of disjunction.