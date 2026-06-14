Logical Predicates
=====================

.. automodule:: jlnn.nn.predicates
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__


This module implements the so-called **Grounding Layers**, which form a semantic interface between the real empirical world (continuous numerical data) and the world of interval fuzzy logic. The task of the predicates is to transform raw features into valid truth intervals :math:`[L, U]`, where :math:`0.0 \le L \le U \le 1.0`.

All predicates inherit from the ``flax.nnx.Module`` base class and are fully compatible with JAX transformations and automatic derivation.

1. Trainable parametric predicate (LearnedPredicate)
--------------------------------------------------------

This predicate independently models and optimizes the lower and upper truth limits for each input feature. Using gradient descent, the network automatically discovers the ideal decision boundaries.

* **Architecture**: Encapsulates four independent learning parameter vectors ``nnx.Param``: ``slope_l``, ``offset_l`` (for the lower bound) and ``slope_u``, ``offset_u`` (for the upper bound).
* **Activation mechanism**: Uses the ``ramp_sigmoid`` function to transform potentials into truth values.
* **Axiomatic fuse**: The resulting tensors are normalized before wrapping using ``intervals.ensure_interval``, which guarantees that the lower bound never exceeds the upper bound, even if the gradients behave chaotically during learning.

.. math::

   L = \text{ramp\_sigmoid}(x \cdot \text{slope}_l - \text{offset}_l)

   U = \text{ramp\_sigmoid}(x \cdot \text{slope}_u - \text{offset}_u)


2. Space-time curved physical predicate (PhysicalPredicate)
---------------------------------------------------------------

An advanced grounding layer that does not use traditional linear displacements but maps real data through a physical fuzzy logic (PFL) topology.

* **Semantics**: Evaluates inputs as physical potentials and decomposes them into a curved space where the mean value :math:`0.5` represents maximum entropy (uncertainty singularity).
* **Configuration**:
   - ``gamma``: Coefficient of the gravitational field intensity of the center of uncertainty.
   - ``mode``: Field compression selection – ``sigmoid`` (continuous field) or ``ramp`` (clipped field).
* **Stabilization**: Excellent at suppressing outliers and anomalies in data by pulling unstable states towards the center of uncertainty proportional to the local Shannon entropy.

3. Static Identity (FixedPredicate)
---------------------------------------

A stateless and completely untrainable predicate that serves as a rigid data channel.

* **Function**: Takes a ready-made truth interval as input and returns it unchanged to the network (Identity Mapping).
* **Usage**: Used for direct injection of pure Boolean facts, expert constants, or deterministic logical anchors into the knowledge graph, which are strictly required to remain completely immune to changes caused by gradient backpropagation.