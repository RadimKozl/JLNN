Reasoning Inspector & Audit
===========================

.. automodule:: jlnn.reasoning.inspector
   :members:
   :show-inheritance:

Module for inspecting and auditing the model. Allows users to look "under the hood" and understand why the model reached a particular conclusion.

Functions for Explainability:
-----------------------------

* **Trace Reasoning**: Recursively traverses the graph and captures activations (intervals) at each node. This is crucial for identifying logical conflicts.
* **Human-Readable Reports**: Funkce ``get_rule_report`` převádí číselné intervaly na přirozený jazyk (např. "True", "Unknown", "Conflict").
