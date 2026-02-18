Meta-Learning & Self-Reflection
==================================

This tutorial demonstrates how to build a **Self-Reflective System** using JLNN. In most neural networks, the training process is a "black box," but JLNN allows us to treat the training history as structured data that can be analyzed by other models.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_xarray_meta-learning.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_xarray_meta-learning.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.


Content Overview
------------------

The pipeline consists of three main stages:

1.  **Logical Reasoning (JLNN):** A model learns rules for "Wine Quality" using weighted logical conjunctions.
2.  **Episodic Memory (Xarray):** Every 25 training steps, the model's internal state (weights, loss, accuracy) is snapshotted into a structured multi-dimensional array.
3.  **Meta-Analysis (Decision Tree):** A Scikit-learn model analyzes this history to discover which parameter settings (like a specific feature weight) were responsible for reaching the highest accuracy.

Core Components
-----------------

Structured Logging with Xarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use ``xarray.Dataset`` to store the training dynamics. This allows us to query the training history using labels instead of raw indices:

.. code-block:: python

    history = xr.Dataset(
        data_vars={
            "member_weights": (["step", "antecedent"], np.zeros((len(steps_to_log), 4))),
            "rule_weight": (["step"], np.zeros(len(steps_to_log))),
            "accuracy": (["step"], np.zeros(len(steps_to_log)))
        },
        coords={"step": steps_to_log, "antecedent": ["alcohol", "acid", "magnesium", "ash"]}
    )

The Meta-Analyst Loop
~~~~~~~~~~~~~~~~~~~~~~~

By training a ``DecisionTreeRegressor`` on the recorded history, we can extract symbolic insights. For example, the tree can tell us: *"If the 'alcohol' weight was above 1.5, accuracy increased by 10%."*

.. code-block:: python

    from sklearn.tree import export_text
    
    # Analyze how weights affected accuracy
    dt_analyzer.fit(df_meta, y_meta)
    print(export_text(dt_analyzer, feature_names=df_meta.columns.tolist()))

Agentic Reflection (Prompt Generation)
----------------------------------------

The ultimate goal of this tutorial is to generate a report that can be understood by an LLM (like Gemma 3 or Llama 3). This output can be fed into an AI agent to autonomously adjust the next training run:

.. code-block:: text

    I analyzed the training of the JLNN model.
    Maximum achieved accuracy: 0.74
    
    Determined logical dependencies:
    |--- magnesium >  1.61
    |   |--- acid <= 0.61 -> value: [0.74]
    
    Recommendation: Focus on magnesium weight stability.

Key Benefits
--------------

* **Transparency:** No more guessing why the model stopped improving.
* **Auto-Tuning:** Foundation for building self-correcting AI agents.
* **Auditability:** A complete trace of the model's "logic evolution" over time.

Tutorial code
---------------

.. code-block:: python

    '''
    try:
        import jlnn
        from flax import nnx
        import jax.numpy as jnp
        import xarray as xr
        import pandas as pd
        import sklearn
        print("✅ JLNN and JAX are ready.")
    except ImportError:
        print("🚀 Installing JLNN from GitHub and fixing JAX for Colab...")
        # Instalace frameworku
        #!pip install jax-lnn --quiet
        !pip install git+https://github.com/RadimKozl/JLNN.git --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        !pip install  scikit-learn pandas

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import os
    os.environ["JAX_PLATFORMS"] = "cpu"

    import os
    import jlnn
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tqdm import tqdm
    import optax
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
    from sklearn.preprocessing import StandardScaler

    data = load_wine()

    X_raw = data.data[:, [10, 1, 4, 3]]
    feature_names = ["alcohol", "acid", "magnesium", "ash"]

    y_raw = (data.target == 0).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    inputs = jnp.stack([X_scaled, X_scaled], axis=-1)
    targets = y_raw[:, None]

    class MetaLogicModel(nnx.Module):
        def __init__(self, n_features, rngs):
            # Weights of individual members (antecedent)
            self.weights = nnx.Param(jnp.ones((n_features,)))
            # Weight of the entire rule (credibility)
            self.rule_weight = nnx.Param(jnp.array([0.9]))

        def __call__(self, x):
            # Implementation of weighted conjunction (Weighted AND)
            w = jnp.abs(self.weights)
            # Simplified Łukasiewicz: norm of the sum of weights
            logic_out = jnp.clip(jnp.sum(w * x[:, :, 0], axis=1) / jnp.sum(w) * self.rule_weight, 0, 1)
            return logic_out

    steps_to_log = np.arange(0, 1000, 25)

    history = xr.Dataset(
        data_vars={
            "member_weights": (["step", "antecedent"], np.zeros((len(steps_to_log), len(feature_names)))),
            "rule_weight": (["step"], np.zeros(len(steps_to_log))),
            "loss": (["step"], np.zeros(len(steps_to_log))),
            "accuracy": (["step"], np.zeros(len(steps_to_log)))
        },
        coords={"step": steps_to_log, "antecedent": feature_names}
    )

    model = MetaLogicModel(len(feature_names), rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    def loss_fn(model, x, y):
        pred = model(x)
        mse = jnp.mean((pred - y[:, 0])**2)
        # Contradiction penalty - penalizes too high weights leading to a contradiction
        return mse + 1.5 * jnp.mean(jnp.maximum(0, jnp.abs(model.weights) - 4.0))

    @nnx.jit
    def train_step(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(model, grads)
        return loss

    print("Starting training and collecting meta-data into Xarray...")

    for i in tqdm(range(1001)):
        # Perform one step of the training
        loss = train_step(model, optimizer, inputs, targets)

        # Data logging at specified intervals
        if i in steps_to_log:
            # 1. Getting antecedent weights (without deprecation warning)
            # .weights[...] returns the current parameter array
            current_weights = np.abs(model.weights[...])

            # 2. Getting the rule weight (fix TypeError)
            # .item() safely converts a single-element array (even with ndim=1) to a Python float
            current_rule_w = model.rule_weight[...].item()

            # 3. Uložení do Xarray Datasetu
            history["member_weights"].loc[dict(step=i)] = current_weights
            history["rule_weight"].loc[dict(step=i)] = current_rule_w
            history["loss"].loc[dict(step=i)] = float(loss)

            # Calculate and store Accuracy
            preds = model(inputs) > 0.5
            acc_value = jnp.mean(preds == y_raw).item() # We'll also use .item() just in case
            history["accuracy"].loc[dict(step=i)] = acc_value

    print("\nTraining complete. Data ready for Meta-Analysis.")

    df_weights = history["member_weights"].to_series().unstack()
    df_meta = df_weights.copy()
    df_meta["rule_weight"] = history["rule_weight"].values

    # Calculate the trend (derivative) of the loss to see learning velocity
    loss_series = history["loss"].to_series()
    df_meta["loss_trend"] = loss_series.diff().fillna(0).values

    y_meta = history["accuracy"].values

    dt_analyzer = DecisionTreeRegressor(max_depth=3)
    dt_analyzer.fit(df_meta, y_meta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Weight development graph
    history.member_weights.plot.line(x="step", ax=ax1)
    ax1.set_title("Evolution of antecedent weights over time")

    # Tree visualization
    plot_tree(dt_analyzer, feature_names=df_meta.columns.tolist(), filled=True, ax=ax2, fontsize=10)
    ax2.set_title("Parameter Analysis: What Determines Model Success?")
    plt.show()

    print("\n" + "="*50)
    print("ANALYSIS FOR LLM AGENT (Gemma/Ollama)")
    print("="*50)

    rules_text = export_text(dt_analyzer, feature_names=df_meta.columns.tolist())
    best_acc = float(y_meta.max()) 

    prompt = f"""
    I analyzed the training of the JLNN model on the Wine Quality data.
    Maximum achieved accuracy: {best_acc:.2f}

    Determined logical dependencies of parameters:
    {rules_text}

    Recommendations for the next iteration:
    1. If the 'alcohol' weight is within the range shown in the top node, focus on stabilizing the 'rule_weight'.
    2. Contradiction penalty 1.5 appears to be optimal for the balance between MSE and weight stability.
    """

    print(prompt)


Download
----------

This tutorial demonstrates how to build a **Self-Reflective System** using JLNN. 
In most neural networks, the training process is a "black box," but JLNN allows us to treat the training history 
as structured data that can be analyzed by other models.

You can also download the raw notebook file for local use:
:download:`JLNN_basic_boolean_gates.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_xarray_meta-learning.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.