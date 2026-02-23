The Grand Cycle: Autonomous Tuning
===================================

This tutorial demonstrates the "Grand Cycle" – a sophisticated optimization loop that automates the setup of a Neuro-Symbolic system.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_grand_cycle_optuna.ipynb
       :link-type: url

       Run the autonomous optimization cycle in your browser.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_grand_cycle_optuna.ipynb
       :link-type: url

       Browse the full source code and results.

Concept: Automated Semantic Discovery
-------------------------------------

In JLNN, we don't just learn weights; we learn the **semantics of the predicates**. This tutorial uses **Optuna** to find the exact fuzzy parameters that make the logical rules most effective.

Key features of this tutorial:

* **Dynamic Grounding:** Automating the ``center`` and ``steepness`` of fuzzy ramps.
* **LNNFormula Integration:** Using high-level logical strings to define the model architecture.
* **Advanced Training:** Utilizing ``optax.warmup_cosine_decay_schedule`` for stable convergence across trials.
* **Agentic Reporting:** Generating a final summary suitable for LLM Agents to drive the next iteration of the model.

How it works
------------

1. **Space Definition:** We define ranges for logical boundaries (e.g., alcohol levels) and training parameters.
2. **Trial Loop:** Optuna runs multiple experiments, each refining the model's understanding of "Wine Quality."
3. **Validation:** Each trial is tested against unseen data to prevent overfitting of the logical rules.
4. **Knowledge Export:** The best parameters are exported as a human-readable report.

Conclusion
----------

The Grand Cycle bridges the gap between traditional AutoML and Symbolic Reasoning, providing a transparent, self-improving AI pipeline.

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
        import optuna
        import matplotlib.pyplot as plt
        import sklearn
        print("✅ JLNN and JAX are ready.")
    except ImportError:
        print("🚀 Installing JLNN from GitHub and fixing JAX for Colab...")
        # Instalace frameworku
        #!pip install jax-lnn --quiet
        !pip install git+https://github.com/RadimKozl/JLNN.git --quiet
        !pip install optuna optuna-dashboard pandas scikit-learn matplotlib
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

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    import optuna
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.utils.xarray_utils import model_to_xarray, extract_weights_to_xarray
    from jlnn.training.losses import total_lnn_loss

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    df["good"] = (df["quality"] >= 7).astype(int)

    features = ["alcohol", "volatile acidity", "sulphates", "chlorides"]
    X = df[features].values
    y = df["good"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)

    def fuzzy_ramp(x, center, steepness):
        return 1 / (1 + jnp.exp(-steepness * (x - center)))

    def create_inputs(X_norm, p):
        high_alcohol   = fuzzy_ramp(X_norm[:, 0], p["c_alcohol"],   p["steepness"])
        low_acidity    = 1 - fuzzy_ramp(X_norm[:, 1], p["c_acidity"],   p["steepness"])
        high_sulphates = fuzzy_ramp(X_norm[:, 2], p["c_sulphates"], p["steepness"])
        low_chlorides  = 1 - fuzzy_ramp(X_norm[:, 3], p["c_chlorides"], p["steepness"])

        # For JLNN – shape (batch, n_literals, 2) for [L,U], here simple [val, val+epsilon]
        epsilon = 0.05
        inputs = {
            "high_alcohol":   jnp.stack([high_alcohol, high_alcohol + epsilon], axis=-1),
            "low_acidity":    jnp.stack([low_acidity,  low_acidity  + epsilon], axis=-1),
            "high_sulphates": jnp.stack([high_sulphates, high_sulphates + epsilon], axis=-1),
            "low_chlorides":  jnp.stack([low_chlorides,  low_chlorides  + epsilon], axis=-1)
        }

        return inputs

    def objective(trial):
        # Hyperparameters to tune
        p = {
            "lr_peak":       trial.suggest_float("lr_peak", 1e-4, 5e-2, log=True),
            "steepness":     trial.suggest_float("steepness", 6.0, 18.0),
            "c_alcohol":     trial.suggest_float("c_alcohol", 0.40, 0.80),
            "c_acidity":     trial.suggest_float("c_acidity", 0.10, 0.45),
            "c_sulphates":   trial.suggest_float("c_sulphates", 0.40, 0.80),
            "c_chlorides":   trial.suggest_float("c_chlorides", 0.05, 0.30),
            "rule_strength": trial.suggest_float("rule_strength", 0.85, 1.00),
            "contra_p":      trial.suggest_float("contra_p", 0.5, 5.0),
        }

        # ─── Creating a model ────────────────────────────────────────────────────
        rule = f"{p['rule_strength']:.3f} :: (high_alcohol & low_acidity & high_sulphates & low_chlorides)"
        model = LNNFormula(rule, nnx.Rngs(42))

        # ─── Optimizer ───────────────────────────────────────────────────────────
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=p["lr_peak"],
            warmup_steps=300,
            decay_steps=8000,
            end_value=1e-6
        )
        tx = optax.adamw(schedule, weight_decay=1e-5)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        # ─── Training ─────────────────────────────────────────────────────────────
        train_inputs = create_inputs(X_train_norm, p)
        target_interval = jnp.where(
            y_train[:, None] == 1,
            jnp.array([[0.85, 1.00]]),
            jnp.array([[0.00, 0.15]])
        )

        best_test_acc = 0.0
        for step in range(3001):

            pred = model(train_inputs)                     # (batch, 4, 2)
            pred_agg = jnp.min(pred, axis=1)               # conservative AND → (batch, 2)
            # or
            # pred_agg = jnp.max(pred, axis=1)             # optimistic
            # pred_agg = jnp.mean(pred, axis=1)            # mean

            mse = jnp.mean((pred_agg - target_interval)**2)
            # or if you have total_lnn_loss which expects (batch, 2):
            loss = total_lnn_loss(pred_agg, target_interval, contradiction_weight=p["contra_p"])

            if step % 1000 == 0:
                test_inputs = create_inputs(X_test_norm, p)
                preds = model(test_inputs)                     # forward pass without grads is the default
                preds_agg = jnp.min(preds, axis=1)             # or max/mean
                test_acc = jnp.mean((preds_agg[:, 0] > 0.5) == y_test)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc

        return best_test_acc

    study = optuna.create_study(direction="maximize", study_name="JLNN_Wine_Grand_Cycle")
    study.optimize(objective, n_trials=35, timeout=2400)  # ~40 min max

    print("\n" + "="*80)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best accuracy test: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:18}: {v:.4f}")
    print("="*80)

    # Charts
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization progress")
    plt.show()

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("The Importance of Hyperparameters\n")
    plt.show()

    optuna.visualization.matplotlib.plot_contour(study, params=["steepness", "rule_strength"])
    plt.title("Steepness vs Rule Strength")
    plt.show()

    report = f"""
    Best achieved test accuracy: {study.best_value:.4f}

    Best configuration:
    {study.best_params}

    Recommendations for the next iteration:
    - Increase the steepness to {study.best_params['steepness'] + 2:.1f}–{study.best_params['steepness'] + 6:.1f}
    - Keep rule_strength close {study.best_params['rule_strength']:.3f}
    - Reduce contradiction penalties under {study.best_params['contra_p']:.1f} if the intervals become too narrow

    Next step: try adding another flag (e.g. 'citric acid') and running Grand Cycle again.
    """

    print("\n" + "="*80)
    print("REPORT FOR LLM AGENT / NEXT ITERATION")
    print("="*80)
    print(report)


Download
----------

You can also download the raw notebook file for local use:
:download:`JLNN_grand_cycle_optuna.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_grand_cycle_optuna.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.