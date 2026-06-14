Functional Logic Kernels
==========================

.. automodule:: jlnn.nn.functional
   :members:
   :undoc-members:
   :show-inheritance:

This module provides a stateless functional interface to all logical operations supported by the JLNN framework. Unlike stateful objects (such as the t-norms in the ``gates`` module), these functions do not maintain any internal parameters or network states. All weights and threshold coefficients are passed to them directly as JAX tensors, making them ideal building blocks for functional programming.

Key features
---------------

* **JAX-Native architecture**: All features are fully compatible with ``jax.jit`` (compilation on XLA/GPU), ``jax.vmap`` (automatic vectorization across batches), and ``jax.grad`` (automatic backpropagation derivation) transformations.
* **Guaranteed Consistency of Bounds**: Each operation internally calls the ``intervals.ensure_interval`` mechanism at the end of the calculation, which enforces strict adherence to the mathematical axiom :math:`L \le U` and prevents the occurrence of negative uncertainty widths.
* **Unified Weighting**: For traditional parametric methods (outside Łukasiewicz logic), the module implements a unified preprocessing contract, where the boundaries of individual operands are adjusted using the product with the appropriate weight and saturated with an upper bound of 1.0.

The core of Łukasiewicz's logic
----------------------------------

These operations work with cumulative evaluation of logical potential and form standard building blocks for LNNs (Logical Neural Networks).

* **weighted_and**: Evaluates the conjunction based on the accumulation of "negative evidence" of the inputs. Uses the parameter ``beta`` as a sensitivity threshold.
* **weighted_or**: Accumulates positive logical evidence across the input tensor.
* **weighted_not**: Performs an interval inversion. First, it applies a weighting coefficient to the boundaries and then performs a logical rotation: :math:`[1.0 - U_w, 1.0 - L_L]`.
* **weighted_nand / weighted_nor**: Compound operators implemented as a functional composition of negation over the result of the corresponding conjunction or disjunction.

Traditional and parametric operators
---------------------------------------

Alternative logical semantics that allow changing the character of the gradient flow and the behavior of the network at the edges of the truth space.

* **weighted_or_kleene_dienes / weighted_and_kleene_dienes**: Max-min operators with weighted inputs, suitable for robust expert systems screening out extreme values.
* **weighted_or_reichenbach / weighted_and_reichenbach**: Smooth probabilistic (product) operators with polynomial nonlinearity, ensuring continuous gradients without sharp breaks.

A complex apparatus of implications
----------------------------------------

The module provides a unified ``implication`` function, which serves as the main router for evaluating rules of the form :math:`A \rightarrow B`. It supports three distinct spheres of computation (``method`` parameter):

1. **Pure atomic implications**:

   - ``implication_lukasiewicz``: Classical nilpotent implication.
   - ``implication_kleene_dienes``: Pessimistic max-not implication.
   - ``implication_reichenbach``: Fully differentiable algebraic implication.
   - ``implication_goguen``: Residual implication using numerically safe division in JAX.
   - ``implication_godel``: Positional residual implication built on conditional masks.

2. **Spacetime-curved PFL implications**:

- ``implication_physical_kleene_dienes``, ``implication_physical_reichenbach``, ``implication_physical_lukasiewicz``. These functions dynamically deform the truth space depending on the local Shannon entropy of both operands. They ensure the stabilization of the system around points of extreme uncertainty.

3. **Weighted parametric implications**:

- If the traditional method is chosen and ``weights`` are passed, a safe transformation of the premise and conclusion boundaries will occur according to the framework's unified weight contract before the actual calculation of the implication.