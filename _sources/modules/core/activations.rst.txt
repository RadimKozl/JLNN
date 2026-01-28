Activation Functions
====================

.. automodule:: jlnn.core.activations
   :members:
   :undoc-members:
   :show-inheritance:

Activation functions in JLNN serve not only for nonlinearity, but also to ensure logical integrity and map real values ​​to truth intervals.

Main functions
--------------

* **identity_activation**: Serves as a numerical safeguard (clipping) to keep values within the range :math:`[0, 1]`.
* **ramp_sigmoid**: Key function for **LearnedPredicates**. Allows conversion of values (e.g., temperature in °C) to logical truth using parameters ``slope`` and ``offset``.

.. math::

   f(x) = \text{clip}(\text{slope} \cdot (x - \text{offset}) + 0.5, 0, 1)

This approach combines the advantages of linearity (stable gradient) with the advantages of saturation (clear definition of absolute truth/falsehood).