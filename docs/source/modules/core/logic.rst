Logical Kernels
==================

.. automodule:: jlnn.core.logic
   :members:
   :undoc-members:
   :show-inheritance:

This module implements low-level mathematical definitions of logical operators operating over truth intervals :math:`[L, U]`. The calculations are designed to strictly preserve interval semantics and correctly propagate uncertainty throughout the network.

All functions are implemented as pure functions optimized for error-free compilation using ``jax.jit``.

1. Łukasiewicz's logic (Nilpotent / Accumulative)
----------------------------------------------------

Standard semantics for JLNN (Logical Neural Networks). Provides stable linear gradients and high interpretability. Conjunction is modeled here through the principle of "negative evidence", where inputs approaching falsely dampen the result.

* **and_lukasiewicz_pure / or_lukasiewicz_pure**: Pure, parametrically unweighted operators.
* **weighted_and_lukasiewicz / weighted_or_lukasiewicz**: Weighted versions controlled by the ``beta`` parameter, which determines the sensitivity or saturation threshold.
* **implies_lukasiewicz**: The implication :math:`A \rightarrow B` modeled as the equivalence :math:`\neg A \lor B`. In interval reasoning, the inversion of limits occurs: :math:`\neg [L, U] = [1 - U, 1 - L]`.

2. Productive logic (Smooth / Probabilistic)
-------------------------------------------------

It models the interactions of independent probabilistic events. Its main feature is a smooth polynomial progression without sharp breaks caused by min/max operators.

* **and_product_pure / or_product_pure**: Basic binary operators.
* **bulk_and_product_raw / bulk_or_product_raw**: N-ary reduction along the last axis of the tensor.
* **implies_reichenbach**: Reichenbach implication defined by the relation :math:`1.0 - A + (A \cdot B)`. It is fully differentiable in the entire unitary domain space, which guarantees stable and non-vanishing gradients for both the premise and conclusion simultaneously.
* **implies_goguen**: Residual R-Implication (Goguen). The calculation uses numerically safe division in JAX to avoid NaN anomalies during backpropagation.

3. Gödel's Logic (Strict / Extremal / Min-Max)
--------------------------------------------------

This logic is based on positional evaluation of extremes. The resulting truth depends purely on the dominant element, not on the cumulative sum.

* **and_godel_pure / or_godel_pure**: Binary min-max operators.
* **bulk_and_godel_raw / bulk_or_godel_raw**: Bulk reductions (vectorized functions ``jnp.min`` and ``jnp.max``).
* **implies_kleene_dienes**: A pessimistic implication model defined as :math:`\max(1.0 - A, B)`. It is highly robust against cumulative error propagation at interval boundaries.
* **implies_godel**: Pure Gödel residue working on the basis of conditional masks ``jnp.where``.

4. Drastic Logic (Theoretical Mathematical Bottom)
------------------------------------------------------

Drastic logic represents the absolute limit barriers of fuzzy logic operators. The t-norm represents the smallest possible t-norm and the t-conorm represents the largest possible.

* **and_drastic_pure**: If one of the components is exactly :math:`1.0`, return the other component, otherwise immediately collapse to pure :math:`0.0`.
* **or_drastic_pure**: If one of the components is equal to :math:`0.0`, return the other, otherwise immediately saturate to the absolute truth :math:`1.0`.
* The module also includes optimized bulk reductions **bulk_and_drastic_raw** and **bulk_or_drastic_raw**, which use precise tolerance masks (the ``EPSILON`` constant) to work with float32 precision.

5. Space-time curved physical fuzzy logic (PFL)
------------------------------------------------------

An advanced PFL (Space-Curved Physical Fuzzy Logic) apparatus that deforms the space of truth potentials based on local entropic chaos (Shannon entropy :math:`H`).

* **implies_physical_kleene_dienes**: Physical implication modulated by local stability of states. Uses stability weights :math:`1.0 - H` to deform the boundaries.
* **implies_physical_reichenbach**: Uses a smooth gravity polynomial :math:`1.0 - A + A \cdot B \cdot (1.0 - H(A) \cdot H(B))`. At maximum system uncertainty (when states approach the center :math:`0.5`), the coupling term decays by eliminating entropy, opening a pure linear channel for stable gradient backpropagation.
* **implies_physical_gravitational_lukasiewicz**: Generates full structural convergence (truth :math:`1.0`) once the logical states enter the entropic singularity region around the point :math:`[0.5, 0.5]`. If the entropy is minimal, it strictly shadows the classical Łukasiewicz behavior.