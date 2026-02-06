Introductory Example: JLNN Base
===============================

This notebook demonstrates the core workflow of JLNN, including rule definition, 
training with contradiction loss, and checkpointing.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/Jax_lnn_base.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/Jax_lnn_base.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.

Content Overview
----------------

In this tutorial, you will learn:

* **Installation**: How to set up the JLNN environment.
* **Symbolic Logic**: Defining rules like ``0.8::A & B -> C``.
* **Grounding**: Transforming raw data into logical truth intervals.
* **Optimization**: Training with ``total_lnn_loss`` and enforcing constraints.
* **Persistence**: Saving and loading model checkpoints.

.. code-block:: python

   '''
   try:
      import jlnn
      from flax import nnx
      import jax.numpy as jnp
      print("✅ JLNN and JAX are ready.")
   except ImportError:
      print("🚀 Installing JLNN from GitHub and fixing JAX for Colab...")
      # Instalace frameworku
      !pip install git+https://github.com/RadimKozl/JLNN.git
      # Fix JAX/CUDA compatibility for 2026 in Colab
      !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

      import os
      print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
      os.kill(os.getpid(), 9)
      os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
   '''
   import os
   os.environ["JAX_PLATFORMS"] = "cpu"

   import jax
   import jax.numpy as jnp
   from flax import nnx
   from jlnn.symbolic.compiler import LNNFormula
   from jlnn.nn.constraints import apply_constraints
   from jlnn.training.losses import total_lnn_loss, logical_mse_loss, contradiction_loss
   from jlnn.storage.checkpoints import save_checkpoint, load_checkpoint
   import optax

   print("JLNN loaded. JAX version:", jax.__version__)

   rngs = nnx.Rngs(42)

   formula = "0.8::A & B -> C"

   model = LNNFormula(formula, rngs)
   print(f"🧩 Model compiled for formula: {formula}")

   inputs = {
      "A": jnp.array([[0.9]]),
      "B": jnp.array([[0.7]]),
      "C": jnp.array([[0.5]])   # MANDATORY – consequent must have grounding!
   }

   target = jnp.array([[0.6, 0.85]])

   initial_pred = model(inputs)
   print(f"Initial prediction (before training): {initial_pred}")

   def loss_fn(model, inputs, target):
      pred = model(inputs)
      pred = jnp.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)  # protection against NaN
      return total_lnn_loss(pred, target)

   optimizer = nnx.Optimizer(
      model,
      wrt=nnx.Param,
      tx=optax.chain(
         optax.clip_by_global_norm(1.0),
         optax.adam(learning_rate=0.001)
      )
   )

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

   final_pred = model(inputs)
   print("\nFinal prediction after training:")
   print(final_pred)

   print(f"\nTarget interval: {target}")
   print(f"Final loss: {total_lnn_loss(final_pred, target):.6f}")


   save_checkpoint(model, "trained_model.ckpt.pkl")
   print("Model saved as trained_model.ckpt.pkl")

   new_model = LNNFormula("0.8::A & B -> C", nnx.Rngs(999))

   load_checkpoint(new_model, "trained_model.ckpt.pkl")
   print("Checkpoint loaded into a new model instance.")

   print("\nPrediction after loading checkpoint:")
   print(new_model(inputs))

   print("\nOriginal prediction (for comparison):")
   print(model(inputs))

Download
--------

You can also download the raw notebook file for local use:
:download:`Jax_lnn_base.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/Jax_lnn_base.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.