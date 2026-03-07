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
   tutorials/index
   theory
   testing

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
----------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from flax import nnx
   from jlnn.symbolic.compiler import LNNFormula
   from jlnn.nn.constraints import apply_constraints
   from jlnn.training.losses import total_lnn_loss, logical_mse_loss, contradiction_loss
   from jlnn.storage.checkpoints import save_checkpoint, load_checkpoint
   import optax

   # 1. Define and compile the formula
   model = LNNFormula("0.8::A & B -> C", nnx.Rngs(42))

   # 2. Ground inputs (including initial state for C)
   inputs = {
      "A": jnp.array([[0.9]]),
      "B": jnp.array([[0.7]]),
      "C": jnp.array([[0.5]])   # MANDATORY – consequent must have grounding!
   }

   target = jnp.array([[0.6, 0.85]])

   # 3. Loss function
   def loss_fn(model, inputs, target):
      pred = model(inputs)
      pred = jnp.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)  # protection against NaN
      return total_lnn_loss(pred, target)

   # 4. Initialize Optimizer
   optimizer = nnx.Optimizer(
      model,
      wrt=nnx.Param,
      tx=optax.chain(
         optax.clip_by_global_norm(1.0),
         optax.adam(learning_rate=0.001)
      )
   )

   # 5. Training Step
   @nnx.jit
   def train_step(model, optimizer, inputs, target):
      # Gradients to the model – closure is traceable (inputs/target are arrays)
      grads = nnx.grad(lambda m: loss_fn(m, inputs, target))(model)

      # Loss before update (for debug)
      loss = loss_fn(model, inputs, target)

      optimizer.update(model, grads)
      apply_constraints(model)

      final_loss = loss_fn(model, inputs, target)
      final_pred = model(inputs)

      return loss, final_loss, final_pred

   print("=== Starting training ===")
   steps = 50
   for step in range(steps):
      loss, final_loss, pred = train_step(model, optimizer, inputs, target)

      print(f"Step {step:3d} | Loss before/after constraints: {loss:.6f} → {final_loss:.6f}")
      print(f"Prediction: {pred}")
      print("─" * 60)

      if jnp.isnan(final_loss).any():
         print("❌ NaN detected! Stopping.")
         break

   print("=== Training completed ===")

   # 6. Result after training

   final_pred = model(inputs)
   print("\nFinal prediction after training:")
   print(final_pred)

   print(f"\nTarget interval: {target}")
   print(f"Final loss: {total_lnn_loss(final_pred, target):.6f}")


**Discord channel:**
~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/URL_QR_Code_def.png
   :align: center
   :width: 250px
   :alt: Discord channel


Indexes
=========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`