Real Example: Iris dataset Classification
===========================================

This tutorial demonstrates the use of JLNN (JAX Logic Neural Network) to find the optimal logical description of the class *Iris Setosa*.
Unlike classical neural networks, the output is learned boundaries ("what is a large leaf")
and logical weights ("which leaf is essential for determining the species").

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_real_world_data_iris.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_real_world_data_iris.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.


Content Overview
------------------

The model learns a neuro-symbolic representation of the rule: *"If the flower is small (short and narrow), then it is an Iris Setosa."*

- **Predicates:** The model transforms real numbers (centimeters) into truth values ​​using learned fuzzy boundaries.
- **Logical operations:** Uses weighted conjunction (Weighted AND) to aggregate attributes.
- **Interpretovatelnost:** Výstupem není jen klasifikace, ale i váhy vyjadřující důležitost jednotlivých vlastností.

Key features of the tutorial
------------------------------

**Integration with Xarray**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the tutorial, we use the ``model_to_xarray`` function, which converts raw JAX output into a structured format with labels. This allows for easy analysis:

.. code-block:: python

    ds = model_to_xarray(
        gate_outputs={"setosa_prediction": preds_agg},
        sample_labels=[f"iris_{i}" for i in range(150)]
    )

**Visualization of learned weights**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The graphical representation of weights w ≥ 1 shows how much the model "listens" to a given input. 
A higher weight for the ``~high_width`` flag means that the width of the ticket 
is more critical to the logical definition of Setosa than its length.

**Uncertainty analysis**
~~~~~~~~~~~~~~~~~~~~~~~~~

JLNN does not just provide a point estimate, but an interval [L, U]. 
The difference U - L defines the uncertainty of the model. In this tutorial, we analyze:

- **Average uncertainty:** How confident the model is across the entire dataset.
- **Uncertainty histogram:** Distribution of the model's "doubts" for individual samples.

**Results and interpretation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, the model achieves high agreement with expert botanical rules. 
The resulting weight graph serves as direct evidence of what the "black box" neural network has learned in the language of symbolic logic.

Example
---------

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
        !pip install jax-lnn --quiet
        #!pip install git+https://github.com/RadimKozl/JLNN.git --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    import optax
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, confusion_matrix

    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.nn.constraints import apply_constraints
    from jlnn.training.losses import total_lnn_loss
    from jlnn.utils.xarray_utils import model_to_xarray, extract_weights_to_xarray

    iris = load_iris()
    X, y = iris.data, iris.target
    # Normalization for logical operations
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)

    def fuzzy_ramp(x, center, steepness=10):
        l = 1 / (1 + jnp.exp(-steepness * (x - (center + 0.1))))
        u = 1 / (1 + jnp.exp(-steepness * (x - (center - 0.1))))
        return jnp.stack([l, u], axis=-1)

    high_length = fuzzy_ramp(X_norm[:, 2], center=0.6)
    high_width  = fuzzy_ramp(X_norm[:, 3], center=0.5)

    unknown_setosa = jnp.ones((len(y), 2), dtype=jnp.float32)
    unknown_setosa = unknown_setosa.at[:, 0].set(0.0) # L=0
    unknown_setosa = unknown_setosa.at[:, 1].set(1.0) # U=1

    target_interval = jnp.where(
        (y == 0)[:, None],
        jnp.array([[0.9, 1.0]]),
        jnp.array([[0.0, 0.1]])
    )

    inputs = {
        "high_length": high_length,
        "high_width": high_width
    }

    formula = "0.9::(~high_length & ~high_width)"
    model = LNNFormula(formula, nnx.Rngs(42))

    optimizer = nnx.Optimizer(model, optax.adam(0.02), wrt=nnx.Param)
    target = (y == 0).astype(jnp.float32)[:, None]

    @nnx.jit
    def train_step(model, optimizer, inputs, target):
        def loss_fn(m):
            pred = m(inputs)
            # If pred returns (150, 2, 2), we reduce the gate dimension to (150, 2)
            if pred.ndim == 3:
                pred = jnp.min(pred, axis=1)
            return total_lnn_loss(pred, target)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        apply_constraints(model)
        return loss

    print("🚀 I'm training a hybrid model (Expert + Data)...")

    # Starting a workout
    for step in range(101):
        loss = train_step(model, optimizer, inputs, target_interval)
        if step % 25 == 0:
            print(f"Step {step:3d} | Loss: {loss:.6f}")

    print("✅ Training ended")

    preds = model(inputs)

    # Safety print – confirm shape
    print("preds shape:", preds.shape)

    # Reduce
    preds_reduced = jnp.min(preds, axis=1)  # (150, 2)
    # preds_reduced = jnp.max(preds, axis=1)   # alternative
    # preds_reduced = jnp.mean(preds, axis=1)

    # Now safe to use
    acc = jnp.mean((preds_reduced[:, 0] > 0.5) == (y == 0))
    print(f"Accuracy: {float(acc):.3f} ({float(acc)*100:.1f}%)")

    # sklearn version (if you prefer)
    acc_sk = accuracy_score(
        (y == 0).astype(int),
        (preds_reduced[:, 0] > 0.5).astype(int)
    )
    print(f"sklearn acc: {acc_sk:.3f}")

    # Uncertainty
    widths = preds_reduced[:, 1] - preds_reduced[:, 0]

    print(f"\n✅ Results:")
    print(f"Accuracy: {acc:.2%}")
    print(f"Average uncertainty (U-L): {float(widths.mean()):.4f}")

    # Assuming your formula has one main conjunction (~high_length & ~high_width)
    # Try to extract directly — the function is designed for this

    da_weights = extract_weights_to_xarray(
        weights=model,  # pass the whole model if it accepts it
        input_labels=["~high_length", "~high_width"],   # or ["not_high_length", "not_high_width"]
        gate_name="conjunction"   # try common names; may need experimentation
    )
    
    # Right after preds = model(inputs)
    print("Original preds shape:", preds.shape)  # confirms (150, 2, 2)

    # Reduce to one [L,U] per sample
    preds_agg = jnp.min(preds, axis=1)           # → (150, 2)
    # Alternatives you can try:
    # preds_agg = jnp.max(preds, axis=1)         # optimistic
    # preds_agg = jnp.mean(preds, axis=1)        # average

    print("Aggregated shape:", preds_agg.shape)  # should be (150, 2)

    ds = model_to_xarray(
        gate_outputs={"setosa_prediction": preds_agg},
        sample_labels=[f"iris_{i}" for i in range(len(y))]
    )

    graphdef, state = nnx.split(model)
    state_dict = state.to_dict() if hasattr(state, 'to_dict') else dict(state)
    weights_var = state_dict['root']['gate']['weights']

    if hasattr(weights_var, 'get_value'):
        conj_weights = weights_var.get_value()
    else:
        conj_weights = weights_var[...]

    
    plt.figure(figsize=(8, 4))

    labels = ["~high_length", "~high_width"]
    values = [float(w) for w in conj_weights.flatten()]

    plt.plot(labels, values, marker='o', linestyle='--', color='teal', linewidth=1.5)
    plt.title("Trained Logic Weights (Importance of Features)")
    plt.ylabel("Weight Value (w >= 1)")
    plt.ylim(0.9, max(values) + 0.5)
    plt.grid(True, alpha=0.3)
    plt.show()


Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_real_world_data_iris.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_real_world_data_iris.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.

