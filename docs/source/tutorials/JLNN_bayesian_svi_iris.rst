Bayesian JLNN: Probabilistic Logic
====================================

This module extends the standard JLNN (JAX Logic Neural Networks) framework 
with the principles of Bayesian statistics. Instead of finding fixed ("point") values ​​for logical boundaries and weights, 
this approach learns entire **probability distributions**.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

Main benefits
---------------
- **Uncertainty quantification:** The model can distinguish between "I know X is true" and "the data suggests X, but I'm not sure".
- **Robustness:** The Bayesian approach naturally penalizes overconfidence on small or noisy datasets.
- **Interpretovatelnost (XAI):** Pomocí intervalů nejvyšší hustoty (HDI) vidíme stabilitu naučených pravidel.

Theoretical basis
-------------------

The module uses **Variational Inference (SVI)** implemented in the `NumPyro` library. 
We approximate a complex posterior P(θ | D) using a simpler distribution Q(θ), 
which we optimize to minimize divergence (ELBO loss).

Logical parameters are modeled as follows:

- **Centers:** Normal(μ, σ) – determines the position of the logical boundary.
- **Steepnesses:** HalfNormal(σ) – determines the sharpness/vagueness of the rule.
- **Weights:** Normal(0, σ) – determines the importance of the rule in the logical sum.

Key visualizations
--------------------

The tutorial documentation includes several essential graphical outputs for model analysis:

1. **HDI Pásy (Sigmoid Nebula):** Vizualizace nejistoty v groundingu jednotlivých predikátů.
2. **Uncertainty Contour:** A 2D surface showing the areas in the feature space where the model is most uncertain.
3. **Posterior Traces:** Diagnostika konvergence parametrů.

Integration with xarray
-------------------------

All inference results are stored in an `xarray.Dataset` object. This allows you to:

- Efficient storage of thousands of posterior samples.
- Easy calculation of statistics (mean, std, percentiles) across `draw`, `rule` and `feature` dimensions.
- Direct export to NetCDF format for archiving logical models.

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
        !pip install numpyro --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
        !pip install  scikit-learn pandas --quiet

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import jax
    import warnings
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
    import optax
    import arviz as az
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    sns.set(style="whitegrid")
    numpyro.set_platform("cpu")

    print(f"JAX Device: {jax.devices()[0]}")

    warnings.filterwarnings("ignore")
    sns.set(style="whitegrid")

    iris = load_iris()
    X_raw = iris.data[:, [2, 3]] # Petal length, Petal width
    y = jnp.array((iris.target == 0).astype(float))
    feature_names = ["petal_length", "petal_width"]

    scaler = StandardScaler()
    X = jnp.array(scaler.fit_transform(X_raw))

    print(f"Data připravena. X_scaled shape: {X.shape}")

    def bayesian_jlnn_model(X, y=None, n_rules=4):
        n_samples, n_features = X.shape

        # Priors using .expand().to_event() for correct shapes
        centers = numpyro.sample("centers",
            dist.Normal(0.0, 1.2).expand([n_rules, n_features]).to_event(2))

        steepnesses = numpyro.sample("steepnesses",
            dist.HalfNormal(10.0).expand([n_rules, n_features]).to_event(2))

        rule_weights = numpyro.sample("rule_weights",
            dist.Normal(0.0, 1.5).expand([n_rules]).to_event(1))

        # Logical Forward Pass
        diff = X[:, None, :] - centers[None, :, :]
        mem = jax.nn.sigmoid(steepnesses[None, :, :] * diff)
        rule_act = jnp.min(mem, axis=-1) # Fuzzy AND
        logits = jnp.sum(rule_act * rule_weights[None, :], axis=-1)

        with numpyro.plate("data", n_samples):
            numpyro.sample("obs", dist.BernoulliLogits(logits), obs=y)


    guide = autoguide.AutoDiagonalNormal(bayesian_jlnn_model)
    svi = SVI(bayesian_jlnn_model, guide, optax.adamw(3e-3), Trace_ELBO(num_particles=10))
    print("Starting training (SVI optimization)...")
    svi_result = svi.run(jax.random.PRNGKey(0), 12000, X, y)

    predictive = Predictive(bayesian_jlnn_model, guide=guide, params=svi_result.params,
                        num_samples=1000, return_sites=["centers", "steepnesses", "rule_weights", "obs"])
    samples = predictive(jax.random.PRNGKey(1), X)

    ds = xr.Dataset({
        "centers": (["draw", "rule", "feature"], samples['centers']),
        "steepnesses": (["draw", "rule", "feature"], samples['steepnesses']),
        "weights": (["draw", "rule"], samples['rule_weights'])
    }, coords={"draw": np.arange(1000), "rule": [f"R{i}" for i in range(4)], "feature": feature_names})

    print("Success: Logical brain model saved to Xarray.")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(svi_result.losses); ax[0].set_title("ELBO Loss"); ax[0].set_yscale('log')

    # Posterior density for steepness (sharpness of the boundary) for Rule 0
    az.plot_posterior(ds.sel(rule="R0", feature="petal_length"), var_names=["steepnesses"], ax=ax[1])
    ax[1].set_title("Posterior Steepness (R0, Petal Length)")
    plt.show()

    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-2.5, 2.5, 100))
    X_grid = jnp.column_stack([xx.ravel(), yy.ravel()])

    # Get predictions for the grid
    # Use return_sites=["obs"], which returns 0/1 (int32)
    grid_predictive = Predictive(bayesian_jlnn_model, guide=guide, params=svi_result.params, num_samples=500)
    grid_samples_raw = grid_predictive(jax.random.PRNGKey(2), X_grid)["obs"]

    # Convert to float32 so JAX can calculate statistics
    grid_samples = grid_samples_raw.astype(jnp.float32)

    # Calculation of average probability and uncertainty (Std Dev)
    # Since these are already Bernoulli samples (0/1), we do not apply sigmoid anymore!
    # The average of zeros and ones will give us the probability (e.g. 0.8 means that in 80% of the samples it was Setosa).
    prob_grid = grid_samples.mean(axis=0).reshape(xx.shape)
    unc_grid = grid_samples.std(axis=0).reshape(xx.shape)

    # Rendering
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Probability map
    c1 = axs[0].contourf(xx, yy, prob_grid, levels=20, cmap='RdBu_r')
    axs[0].scatter(X[:,0], X[:,1], c=y, edgecolors='k', alpha=0.5)
    axs[0].set_title("Mean Probability P(Setosa) - Posterior Predictive")
    fig.colorbar(c1, ax=axs[0])

    # Uncertainty map (where the model hesitates)
    c2 = axs[1].contourf(xx, yy, unc_grid, levels=20, cmap='viridis')
    axs[1].scatter(X[:,0], X[:,1], c='white', edgecolors='k', s=20, alpha=0.3)
    axs[1].set_title("Epistemic Uncertainty (Standard Deviation)")
    fig.colorbar(c2, ax=axs[1])

    plt.show()

    def print_hdi_report(ds):
        print("--- BAYESIAN LOGIC REPORT ---")
        for r in ds.rule.values[:2]: # For clarity, only the first two rules
            for f in ds.feature.values:
                s_vals = ds.steepnesses.sel(rule=r, feature=f).values
                hdi = az.hdi(s_vals, hdi_prob=0.90)
                mean_s = s_vals.mean()

                uncertainty = hdi[1] - hdi[0]
                status = "VYSOKÁ JISTOTA" if uncertainty < 5 else "HESITATION / VAGUE"
                print(f"[{r} | {f}]: Steepness {mean_s:.1f} | 90% HDI: [{hdi[0]:.1f} - {hdi[1]:.1f}] -> {status}")

    print_hdi_report(ds)

    plt.figure(figsize=(8, 2))
    sns.heatmap(ds.weights.mean(dim="draw").to_pandas().to_frame().T, annot=True, cmap="coolwarm")
    plt.title("Mean Posterior Rule Weights (Rule Influence)")
    plt.show()


Download
----------

You can also download the raw notebook file for local use:
:download:`JLNN_bayesian_svi_iris.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_bayesian_svi_iris.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.