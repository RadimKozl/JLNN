Bayesian JLNN: Logic in an Uncertain World
============================================

This tutorial presents the technological pinnacle of the **JLNN (Joint Logic Neural Network)** framework. While standard models try to "fit" the world into fixed boundaries, the Bayesian variant acknowledges reality: **data is noisy and the world is full of uncertainty**.

Instead of finding fixed point values ​​for logical boundaries and rule weights, this approach learns entire **probability distributions** (Posteriors).

.. note::
    The interactive notebook is hosted externally, allowing for immediate launch in the cloud without the need for local installation.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

Main benefits (LNN & BNN)
---------------------------
- **Confidence over Certainty:** The model can identify both sharp logical boundaries (narrow HDI) and vague areas where the rules encounter noise in the data.
- **Explainable Uncertainty:** If the model is uncertain, the distribution of rule weights (``w_rules``) tells us whether the problem is conflicting observations or lack of evidence.
- **Safety-First AI:** The output is not just a number, but a probability distribution. This is key for critical applications where the answer "I don't know" is more valuable than a wrong guess.
- **Seamless Integration:** Full integration of the JLNN symbolic compiler with the NumPyro library and the JAX ecosystem.

Technical pillars of implementation
--------------------------------------

1. **LNNFormula Compiler:**
Automatic conversion of text rules (e.g. ``"petal_length > 2.5 -> Virginica"``) into a computational graph built on Flax NNX.

2. **Stochastic Variable Grounding:**
Predicate parameters (slope and shift of the logical sigmoid) are transformed into latent variables in the NumPyro model.

3. **Stochastic Variational Inference (SVI):**
Instead of the laborious MCMC, we use SVI to approximate the posterior, which allows us to scale logical reasoning to more complex problems.

4. **Xarray & ArviZ Integration:**
The results are not just numbers, but named multidimensional datasets ready for meta-learning and automatic model reflection.


Theoretical basis
-------------------

The module uses **Variational Inference (SVI)** implemented in the `NumPyro` library. 
We approximate a complex posterior P(θ | D) using a simpler distribution Q(θ), 
which we optimize to minimize divergence (ELBO loss).

Logical parameters are modeled as follows:

- **Centers:** Normal(μ, σ) – determines the position of the logical boundary.
- **Steepnesses:** HalfNormal(σ) – determines the sharpness/vagueness of the rule.
- **Weights:** Normal(0, σ) – determines the importance of the rule in the logical sum.


Integration with xarray
-------------------------

All inference results are stored in an ``xarray.Dataset`` object. This allows:

- Efficient storage of thousands of posterior samples.
- Easy calculation of statistics (mean, std, percentiles) across ``draw``, ``rule`` and ``feature`` dimensions.
- Direct export to NetCDF format for archiving logical model states.

Key Outputs
-------------

1. **SVI Convergence (ELBO):** Diagnostics of loss function decline to verify successful posterior approximation.
2. **Rule Weights (Credible Intervals):** Forest plot showing confidence intervals for rule weights. If the interval crosses zero, the rule has no support in the data in the given context.
3. **Epistemic Uncertainty Tracking:** Quantification of uncertainty for individual samples, allowing for in-depth auditing of the decision process.

Tutorial code
---------------

.. code-block:: python

    '''
    try:
        import jlnn
        import numpyro
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
        !pip install optuna optuna-dashboard pandas scikit-learn matplotlib --quiet
        !pip install numpyro jraph arviz --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
        !pip install  scikit-learn pandas --quiet

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import warnings
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import arviz as az
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
    from flax import nnx

    # Imports from JLNN
    from jlnn.symbolic.compiler import LNNFormula

    jax.config.update("jax_enable_x64", True)

    warnings.filterwarnings("ignore")
    sns.set(style="whitegrid")
    numpyro.set_platform("cpu")

    print(f"JAX Device: {jax.devices()[0]}")

    iris = load_iris()

    X = iris.data[:, [2, 3]].astype(jnp.float64)
    y = (iris.target == 0).astype(jnp.float64)  # Is Setosa?

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rule_formulas = [
        "(petal_length_small & petal_width_small) -> is_setosa",
        "petal_length_large -> ~is_setosa"
    ]

    n_rules = len(rule_formulas)
    n_features = X.shape[1]

    rngs = nnx.Rngs(42)
    rule_models = [LNNFormula(f, rngs) for f in rule_formulas]
    n_rules = len(rule_models)
    n_features = X.shape[1]

    print("✅ Symbolic rules compiled:")
    for i, f in enumerate(rule_formulas):
        print(f"  R{i}: {f}")

    def ramp_sigmoid(x, slope, offset):
        """Grounding function converting the input to fuzzy truth [0, 1]."""
        return jax.nn.sigmoid(slope * (x - offset))

    def bayesian_jlnn_model(X_data, y_obs=None):
        n_samples, n_feats = X_data.shape

        # Predicate parameters (LNN Grounding)
        # We use plate to vectorize parameters across rules and flags
        with numpyro.plate("rules_plate", n_rules, dim=-2):
            with numpyro.plate("features_plate", n_feats, dim=-1):
                s_l = numpyro.sample("s_l", dist.HalfNormal(5.0))
                o_l = numpyro.sample("o_l", dist.Normal(0.0, 1.0))
                s_u = numpyro.sample("s_u", dist.HalfNormal(5.0))
                o_u = numpyro.sample("o_u", dist.Normal(0.0, 1.0))

                # Deterministic sites allow Predictive to pull these values ​​into the posterior
                numpyro.deterministic("slope_l", s_l)
                numpyro.deterministic("offset_l", o_l)

        # Rule weights (Importance of each rule)
        with numpyro.plate("weights_plate", n_rules):
            w = numpyro.sample("w", dist.Normal(1.0, 0.5))
            numpyro.deterministic("w_rules", w)

        # --- VECTORIZED LOGICAL INFERENCE ---
        # L, U represent the lower and upper bounds of truth
        L = ramp_sigmoid(X_data[:, None, :], s_l[None, :, :], o_l[None, :, :])
        U = ramp_sigmoid(X_data[:, None, :], s_u[None, :, :], o_u[None, :, :])

        # Aggregation: T-norm AND (minimum) over features
        rule_activations = jnp.min(L, axis=-1)

        # Final prediction (Logit combination)
        logits = jnp.sum(rule_activations * w[None, :], axis=-1)
        numpyro.deterministic("logits", logits)

        # Tracking logical contradictions
        contra = jnp.mean(jnp.maximum(0, L - U))
        numpyro.deterministic("logical_contradiction", contra)

        # Observation (Likelihood)
        with numpyro.plate("data", n_samples):
            numpyro.sample("obs", dist.BernoulliLogits(logits), obs=y_obs)
    
    print("🚀 Running SVI optimization (Stochastic Variational Inference)...")

    guide = autoguide.AutoDiagonalNormal(bayesian_jlnn_model)
    optimizer = numpyro.optim.Adam(step_size=0.005)
    svi = SVI(bayesian_jlnn_model, guide, optimizer, loss=Trace_ELBO())

    print("Starting SVI optimisation (12 000 steps) …")
    svi_result = svi.run(jax.random.PRNGKey(42), 12_000, X_train, y_train)
    params = svi_result.params
    print(f"Final ELBO loss: {svi_result.losses[-1]:.4f}")

    print("📊 I generate posterior samples for uncertainty analysis...")
    predictive = Predictive(bayesian_jlnn_model, guide=guide, params=params, num_samples=1000)
    posterior_samples = predictive(jax.random.PRNGKey(123), X_data=X_test)

    idata = az.from_dict(
        posterior={
            "slope_l": posterior_samples["slope_l"][None, ...],
            "w_rules": posterior_samples["w_rules"][None, ...]
        },
        observed_data={"y": y_test}
    )

    plt.figure(figsize=(8, 4))
    plt.plot(svi_result.losses)
    plt.title("SVI Convergence (ELBO)")
    plt.yscale("log")
    plt.xlabel("Iteration") 
    plt.ylabel("Loss")
    plt.show()

    print("\n✅ Statistical overview of rule weights (Interpretability):")
    summary = az.summary(idata, var_names=["w_rules"], stat_focus="mean")
    print(summary)

    az.plot_forest(idata, var_names=["w_rules"], combined=True, figsize=(8, 4))
    plt.title("Rule Weights: Credible Intervals")
    plt.axvline(0, color='r', linestyle='--')
    plt.show()

    probs = jax.nn.sigmoid(posterior_samples["logits"])
    mean_pred = probs.mean(axis=0)
    uncertainty = probs.std(axis=0)

    accuracy = jnp.mean((mean_pred > 0.5) == y_test)
    print(f"\n🎯 Accuracy on the test set: {accuracy:.2%}")
    print(f"⚠️ Average logical contradiction: {jnp.mean(posterior_samples['logical_contradiction']):.6f}")

    # Show uncertainty for the first 5 samples
    print("\n🔍 Epistemic Uncertainty:")
    for i in range(5):
        print(f"Sample {i}: Prediction={mean_pred[i]:.4f}, Uncertainty (std)={uncertainty[i]:.4f}")


Conclusion
------------

Bayesian JLNN changes the way we think about AI. Instead of a black box that is always sure, you get a partner that understands logic and is not afraid to admit doubt where the data ends.


Download
----------

You can also download the raw notebook file for local use:
:download:`JLNN_bayesian_svi_iris.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.