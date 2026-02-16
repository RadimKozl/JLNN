Changelog
==========

All significant changes to the JLNN project will be documented in this file. The project adheres to semantic versioning.

.. note::
   JLNN is currently in the **Alpha** phase. API may change based on feedback from research deployments.

[0.1.rc2] - 2026-02-15
-----------------------

This Release Candidate brings a major fix to the export pipeline and refines the core logical consistency of the framework.

**Added**
^^^^^^^^^^
* **FixedPredicate:** Introduced a non-trainable identity predicate specifically for crisp (exact 0/1) logic examples.
* **Robust PyTree Support:** The export pipeline now fully supports dictionary-based predicate inputs (e.g., `{"A": tensor, "B": tensor}`) using `jax.tree.map`.
* **Metadata Resolution:** Added `get_representative_shape` helper to ensure correct ONNX `value_info` metadata generation even for complex nested inputs.

**Changed**
^^^^^^^^^^^^

* **Export Pipeline Refactoring:** Updated `export_to_stablehlo` and `export_to_onnx` to handle structured PyTrees instead of requiring flat `jnp.ndarray` inputs.
* **Negation Axiom (weighted_not):** Corrected the order of operations to apply pure negation ([1-U, 1-L]) before weight scaling.
* **Workflow Consistency:** The `export_workflow_example` now demonstrates end-to-end StableHLO and ONNX export using dictionary-based inputs.
* **Interval Enforcement:** Ensured consistent L≤U after every logical operation via the `ensure_interval` mechanism.

**Fixed**
^^^^^^^^^^

* **AttributeError in Export:** Resolved the critical error where the exporter attempted to call `.shape` on dictionary objects.
* **AttributeError in Visualization:** Fixed key handling for fuzzy inputs within visualization loops.
* **Crisp Negation Logic:** Fixed incorrect output values for crisp negation (e.g., :math:`\sim 0 \rightarrow [1,1]` and :math:`\sim 1 \rightarrow [0,0]`).
* **Uncertainty Preservation:** Negation now correctly transfers uncertainty widths in fuzzy inputs (e.g., :math:`[0.95, 1.0] \rightarrow [0.0, 0.05]`).


[0.1.rc1] - 2026-02-10
-----------------------

Release Candidate 1 – significant improvements in negation logic, crisp mode support, and documentation.

**Added**
^^^^^^^^^^
* **FixedPredicate** – non-trainable identity predicate for crisp (exact 0/1) logic examples
* **Two new tutorials**:
  - Introductory Example (training + checkpoint demo)
  - Basic Boolean Gates (AND/OR/NOT/NAND/NOR/XOR with crisp vs fuzzy comparison)
* Visualization of uncertainty propagation (width U-L) in boolean operations
* Support for crisp logic mode (bypassing LearnedPredicate ramps)

**Changed**
^^^^^^^^^^^^
* **weighted_not** – corrected negation axiom: pure negation first ([1-U, 1-L]), then weight scaling
* Ensured consistent L ≤ U after every logical operation (via ensure_interval)
* Improved documentation structure with dedicated Tutorials section

**Fixed**
^^^^^^^^^^
* Negation width preservation – fuzzy inputs now correctly transfer uncertainty (e.g. [0.95,1.0] → ~ = [0.0,0.05])
* AttributeError in visualization loop (fixed key handling for fuzzy inputs)
* Incorrect output for crisp negation (~0 → [1,1], ~1 → [0,0])

[0.1.0] - 2026-02-06
---------------------
First public release (Alpha Release). Implementation of the core framework built on JAX and Flax NNX.

Added
^^^^^^^
* **Example Notebooks**: Added Jupyter notebooks demonstrating basic usage, training, and reasoning with JLNN.
* **README**: Created a comprehensive README with installation instructions and quickstart guide.

Changed
^^^^^^^
* Updated documentation structure to include user guide and tutorials.

[0.0.2] - 2026-01-27
---------------------

First public release (Alpha Release). Implementation of the core framework built on JAX and Flax NNX.

Added
^^^^^^^
* **Core Engine**: Implementation of Łukasiewicz logic for interval arithmetic :math:`[L, U]`.
* **Symbolic Compiler**: Parser for logical formulas (Lark) and automatic compiler to hierarchical NNX modules.
* **Temporal Logic**: Support for operators *Always* (G) and *Eventually* (F) for time series analysis.
* **Constraint System**: Mechanismus ``Projected Gradient Descent`` for maintaining weights :math:`w \ge 1`.
* **Export Pipeline**: Ability to export models to **StableHLO** and **ONNX** formats.
* **Visualizer**: Tools for visualizing truth intervals and gate weights using Matplotlib and Seaborn.
* **Xarray Integration**: Support for scientific data formats and labeled outputs.

Changed
^^^^^^^
* Created on the basis of modern **Flax NNX**.
* Optimization of recursive graph traversal for full compatibility with ``jax.jit``.

Fixed
^^^^^^^
* Fix numeric instability in XOR gate when intervals are very narrow.
* Add base documentation.

[0.0.1] - 2026-01-17
---------------------
* Create a project
* Basic framework concept