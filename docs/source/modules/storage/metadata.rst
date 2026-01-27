Model Metadata
==============

.. automodule:: jlnn.storage.metadata
   :members:
   :show-inheritance:

Metadata in JLNN serves as a bridge between the numerical world of neural networks and the symbolic world of logic. Without metadata, learned weights would be just a list of numbers without knowledge of which predicate (e.g., "Temperature") they belong to.

Metadata Structure:
-------------------

A typical ``.json`` file contains:
* **predicate_names**: List of names of input sensors/facts.
* **logic_semantics**: Information about the used semantics (e.g., 'lukasiewicz').
* **version**: Model version for tracking experiments.

.. autofunction:: save_metadata
.. autofunction:: load_metadata