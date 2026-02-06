Quickstart
==========

This guide follows the introductory example from our :doc:`tutorials/introduction_tutorial`. It demonstrates how to compile a logical formula, perform inference with intervals, and train the model using JAX.

1. Installation
----------------

To get started in a local environment or Colab, install directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/RadimKozl/JLNN.git

2. Define and Compile Logic
----------------------------

The :class:`LNNFormula` takes a string formula and creates a differentiable graph of Flax NNX modules.

.. code-block:: python

    from jlnn.symbolic import LNNFormula
    from flax import nnx
    import jax.numpy as jnp

    rngs = nnx.Rngs(42)
    # Using the rule from the introductory example
    formula = "0.8::A & B -> C"
    model = LNNFormula(formula, rngs)

3. Inference with Intervals
----------------------------

JLNN uses truth intervals $[L, U]$. Even the conclusion ``C`` requires a grounding input (e.g., set to an uncertain state ``[0, 1]``).

.. code-block:: python

    # Define inputs for A, B, and the initial state of C
    inputs = {
        "A": jnp.array([[1.0]]), # Certainly True
        "B": jnp.array([[1.0]]), # Certainly True
        "C": jnp.array([[0.0]])  # Initial grounding
    }

    # Forward pass returns the [Lower, Upper] interval
    prediction = model(inputs)
    print(f"Prediction for C: {prediction}")

4. Training (NaN-free)
-----------------------

To train the model, we use ``optax`` and the specialized :func:`jlnn_learning_loss` which handles MSE, contradictions, and uncertainty.

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

Next Steps
-----------