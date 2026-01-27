.. JLNN documentation master file, created by
   sphinx-quickstart on Tue Jan 27 08:53:06 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

JLNN: JAX Logical Neural Networks
=========================================

**JLNN** is a high-performance neuro-symbolic framework built on a modern JAX stack and **Flax NNX**. It allows you to define logical knowledge using human-readable formulas and then compile them into differentiable neural graphs.

.. image:: _static/jlnn_diagram.png
   :align: center
   :alt: JLNN Architecture Diagram

Why JLNN?
----------

Unlike standard neural networks, JLNN works with **interval logic** (truth is not just a point, but a range $[L, U]$). Thanks to this, the framework can detect not only what it "knows", but also where the data is contradictory (**Contradiction**) or where it lacks information (**Uncertainty**).

Key Components
------------------

* **Symbolic Compiler**: Using Lark grammar, transforms string definitions (e.g. ``A & B -> C``) directly into the NNX module hierarchy.
* **Graph-Based Architecture (NetworkX)**: Full support for bidirectional conversion between JLNN and NetworkX. Allows importing topology from graph databases and visualizing logical trees as hierarchical graphs using ``build_networkx_graph``.
* **Flax NNX Integration**: Uses the latest state management in Flax, ensuring lightning speed, clean parameter handling, and compatibility with XLA.
* **Constraint Enforcement**: Built-in projected gradients ensure that the learned weights :math:`w \geq 1` always conform to logical axioms.
* **Unified Export**: Direct path from trained model to **ONNX**, **StableHLO** and **PyTorch** formats.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/core/index
   modules/export/index
   modules/nn/index
   modules/reasoning/index
   modules/storage/index
   modules/symbolic/index
   modules/training/index
   modules/utils/index

.. toctree::
   :maxdepth: 1
   :caption: About the Project:

   license
   contributing
   changelog

Example of use
---------------

.. code-block:: python

   # code example


Indexes
=========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`