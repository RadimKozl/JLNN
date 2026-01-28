Theoretical foundations of JLNN
=======================================

JLNN combines classical propositional logic with deep learning using three pillars:

1. Interval truth
-------------------------
Unlike standard fuzzy systems, where the truth value is :math:`a \in [0, 1]`, JLNN works with the interval:

.. math::

   I_a = [L_a, U_a], \quad 0 \leq L_a \leq U_a \leq 1

* **Contradiction**: If :math:`L_a > U_a` occurs during learning, the system detects a logical contradiction.
* **Uncertainty**: The width of the interval :math:`U_a - L_a` represents the degree of ignorance (epistemic uncertainty).

2. Weighted Łukasiewicz's logic
-------------------------------
The operators (AND, OR, ..., IMPLY) are implemented as differentiable functions. For an AND gate with weights :math:`w_i` and threshold :math:`\beta`:

.. math::

   L_{out} = \max\left(0, 1 - \sum w_i (1 - L_i) / \beta\right)

Weights :math:`w \geq 1` allow the model to learn the relevance of individual antecedents.

3. Mapping to graphs
--------------------
Thanks to the integration with **NetworkX**, each logical formula is represented as a directed acyclic graph (DAG).

* **Leaves**: PredicateNodes (inputs).
* **Internal nodes**: Logical Gates (operations).
* **Edges**: Flow of truth intervals.

This structure allows you to export the model to ``.dot`` format for visualization or to execute graph algorithms directly on top of the logical rule structure.