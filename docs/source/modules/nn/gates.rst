Logical Gates (Stateful)
==========================

.. automodule:: jlnn.nn.gates
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

This module contains an implementation of logical gates implemented as state objects (modules) in the modern **Flax NNX** framework. Each parametric gate encapsulates its own trainable parameters (input importance weights and activation thresholds) that it optimizes using gradient descent, while physical gates model nonlinear truth space deformations.

Stateful Gates vs. Stateless Functions
-----------------------------------------

Unlike the functional interface in the :doc:`functional` module, the gateways in this module:

* They store their internal state (weight and threshold matrices) by encapsulating them in ``nnx.Param``.
* They are fully compatible with Flax NNX array injection mechanisms and automatic parameter search using optimizers.
* They serve as basic building blocks that the compiler assembles into deep neuro-symbolic structures and knowledge graphs.
* To access tensor values ​​inside JAX operations, they strictly use the modern NNX syntax with ellipsis: ``self.weights[...]`` and ``self.beta[...]``.

1. Traditional parametric logic gates
-----------------------------------------

These gates implement learning logical operators over truth intervals. Using weights, the network can selectively suppress irrelevant inputs and use the ``beta`` (bias) parameter to adjust the stringency of the logical evaluation.

* **WeightedAnd**: Implements weighted conjunction. By default, it uses the Łukasiewicz t-norm based on the accumulation of "negative evidence" of inputs. It also supports Kleene-Dienes and Reichenbach semantics.
* **WeightedOr**: Implements weighted disjunction (t-conorm). In the Łukasiewicz variant, the ``beta`` parameter determines how much positive evidence is needed to saturate to the absolute truth of 1.0.
* **WeightedNot**: Single-input trainable negation, where the weighting coefficient defines the degree (stringency) of the interval inversion.
* **WeightedImplication**: Encapsulates causal rules of type :math:`A \rightarrow B`. Allows training independent weights for the premise and conclusion of the rule.
* **WeightedXor**: N-ary exclusive disjunction implemented as a hierarchical tree of binary XOR operations. This structure allows for efficient learning of complex parity functions.
* **Compound gates (WeightedNand, WeightedNor)**: Combination of weighted operators followed by negation. Key for detecting logical contradictions and penalizing violations of knowledge base integrity.

2. Bulk Gates
------------------

Bulk operators are used to aggregate large tensor structures along a specified dimension (the ``axis`` parameter), which is ideal for evaluating quantifiers (general :math:`\forall` and existential :math:`\exists`) over vector fields.

* **BulkAnd**: Bulk conjunction reducing the selected tensor axis using Gödel's minimum or Product (probabilistic) operator.
* **BulkOr**: Bulk disjunction evaluating the maximum or probabilistic union along a specified axis.

3. Space-time-curved physics gates (PFL)
-------------------------------------------

These gates represent parameter-free modules that, instead of learning weights, transform logical operations through topological curvature of the truth field proportional to the local Shannon entropy.

* **PhysicalAnd / PhysicalOr**: Evaluates conjunction and disjunction in curved space. The ``gamma`` parameter determines the intensity of the gravitational field of the center of uncertainty (0.5) that attracts unstable states.
* **PhysicalImplication**: Pure physical mapping of causal relationships. Supports advanced PFL semantics (``physical_kleene_dienes``, ``physical_reichenbach``, ``physical_lukasiewicz``) that eliminate vanishing gradients in regions of high system entropy.
* **PhysicalNot / PhysicalNand / PhysicalNor**: Physical invariants for inversion and composite negation working with topological fields without the need to store internal weights in GPU memory.
