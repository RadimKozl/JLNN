Activation Functions
=======================

.. automodule:: jlnn.core.activations
   :members:
   :undoc-members:
   :show-inheritance:

Activating a function within the JLNN (Logical Neural Networks) framework not only serves to introduce nonlinearity into the network, but also fulfills two key semantic tasks: ensuring strict preservation of logical integrity and mapping real inputs or hidden potentials into correct truth intervals :math:`[0, 1]`.

Traditional parametric activation
------------------------------------

These functions ensure saturation at the edges and linearity in the gradient region, which is essential for stable and interpretable learning of gate and predicate weights.

* **identity_activation**: Serves as a numeric safety (clipping with ``jnp.clip``) to keep values ​​within the bounds :math:`[0, 1]`. Prevents overflows caused by cumulative rounding when operating on tensors.
* **ramp_sigmoid**: A key function for **LearnedPredicates**. It allows the conversion of real physical quantities to fuzzy logical truth using the parameters ``slope`` (stringency) and ``offset`` (decision boundary).

.. math::

   f(x) = \text{clip}(\text{slope} \cdot (x - \text{offset}) + 0.5, 0, 1)

Activation for Łukasiewicz logic
-----------------------------------

They implement specific accumulation and nilpotent activations for weighted logic gates, where the parameter ``beta`` (:math:`\beta`) represents the sensitivity threshold.

* **lukasiewicz_and_activation**: Transforms the weighted sum of the complements (negations) of the inputs into the resulting truth value of the conjunction. If the sum of the "falsehoods" exceeds the value :math:`\beta`, the output drops to pure zero.

.. math::

   f(s, \beta) = \text{clip}\left(1.0 - \frac{s}{\beta}, 0, 1\right)


* **lukasiewicz_or_activation**: Activation for disjunction. The lower the value of :math:`\beta`, the fewer "confirmations" (sum of the truth values ​​of the inputs) are needed to reach absolute truth.

.. math::

   f(s, \beta) = \text{clip} \left(\frac{s}{\beta}, 0, 1\right)


Physical fuzzy logic (PFL) and space curvature
-------------------------------------------------

Extension of the framework to include space-time curved relations (Space-Curved Physical Fuzzy Logic), which model local chaos and state stability using Shannon entropy.

* **entropy_raw**: Computes the normalized binary Shannon entropy on the interval :math:`[0, 1]`. It reaches a maximum at :math:`H(0.5) = 1.0` (pure uncertainty) and falls to :math:`H(0.0) = H(1.0) = 0.0` at deterministic edges.

.. math::

   H(v) = -v \log_2(v) - (1-v) \log_2(1-v)


* **get_entropic_weight**: Determines the logical stability of the state as a complement of entropy: :math:`1.0 - H(v)`. Used to dynamically adjust the limits of physical implication operators.
* **gravitational_bend_activation**: Complex PFL activation that warps the standard truth potential space. Simulates a gravitational well around the center of uncertainty (:math:`0.5`) proportional to the local entropy and the strength of the ``gamma`` coefficient. It attracts unstable states to the center, while at the deterministic edges the influence of gravity naturally disappears.

It supports two compression modes (``mode`` parameter):

- ``sigmoid``: Smooth, continuous simulation of a physical field.
- ``ramp``: Truncated lineární mapování s ostřejšími přechody za použití parametrů ``slope`` a ``offset``.