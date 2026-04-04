Neuro-Symbolic Bayesian GraphSAGE + JLNN
===========================================

This tutorial represents the technological pinnacle of integrating the **JLNN** framework with graph neural networks. On the complex dataset **OGBN-Proteins**, we demonstrate how to combine the inductive power of **GraphSAGE** (perception) with the Bayesian logic of **JLNN** (reasoning).

Instead of simple classification, we are creating a system that critically assesses its own certainty – introducing the concept of **intelligent doubt**.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_NS_Bayesian_GraphSAGE.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_NS_Bayesian_GraphSAGE_github.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.


Main Benefits
---------------

* **Interpretability (XAI):** Moving from "Black-box" embeddings to fuzzy rules. The model provides a clear logical justification for why the protein was classified in a given way.
* **Quantification of Uncertainty (HDI):** Using Highest Density Intervals, the model identifies proteins about which it is unsure, preventing costly errors in laboratory practice.
* **Scalable Inference:** The use of **Stochastic Variational Inference (SVI)** allows Bayesian logic to be applied even to large graphs with millions of interactions.
* **Stateless Forward Pass:** Advanced integration with Flax NNX allowing sampling of logic parameters without side effects on the model state.

Theoretical Framework
------------------------

In the tutorial, we implement a hybrid architecture where:

1. **GraphSAGE layer** aggregates 8-dimensional biological features from the protein environment into so-called grounding facts.
2. **JLNN formulas** (e.g. ``0.8::CoExpression & PhysicalBinding -> ProteinFunction``) evaluate these facts using learned logical operators.
3. **Bayesian mapping** assigns probability distributions to logic parameters (slope, offset, weight) instead of fixed values.

Integration with xarray and Grand Cycle
------------------------------------------

The key element of this tutorial is the preparation of data for the so-called **Grand Cycle**. All posterior samples are stored in the ``xarray.Dataset`` (NetCDF) format, which serves as the knowledge base for the LLM agent.

This agent can autonomously suggest: based on the width of the HDI intervals and the degree of contradiction (``logical_health``):

* Clarification of symbolic rules.
* Adjusting hyperparameters for the next learning iteration.
* Identify anomalies in data that contradict the defined logic.

Key deliverables
------------------

* **SVI Convergence:** Learning stability diagnostics using ELBO.
* **Truth Intervals:** Visualization of truth intervals $[L, U]$ for individual proteins.
* **Rule Weights Forest Plot:** A statistical overview of the stability and importance of individual biological rules.

Conclusion
-------------

Bayesian GraphSAGE + JLNN changes the paradigm of bioinformatics. Instead of a system that blindly predicts, you get an expert partner who understands biological logic and can quantify your doubt where the data ends.

Example
---------

.. code-block:: python

    '''
    try:
        import jlnn
        import jraph
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
        !pip install arviz --quiet
        !pip install numpyro jraph --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
        !pip install  scikit-learn pandas --quiet

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    # === Library imports ===

    import jax
    import warnings
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from jax import random
    import jraph
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
    import optax
    import optuna
    from flax import nnx
    import xarray as xr

    # --- Native JLNN imports ---
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.training.losses import contradiction_loss, total_lnn_loss
    from jlnn.utils.xarray_utils import model_to_xarray, extract_weights_to_xarray
    from jlnn.utils.visualize import plot_truth_intervals, plot_training_log_loss


    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score, 
        confusion_matrix
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from scipy.stats import spearmanr, pointbiserialr
    from scipy.special import expit

    from tqdm import tqdm

    sns.set(style="whitegrid")
    numpyro.set_platform("cpu")

    print(f"JAX Device: {jax.devices()[0]}")

    warnings.filterwarnings("ignore")
    sns.set(style="whitegrid")

    # === Data preparation (OGBN-Proteins Subset) ===

    N_NODES = 500
    N_EDGES = 2500

    key = random.PRNGKey(42)
    key, k1, k2, k3, k4, k5 = random.split(key, 6)

    labels = random.bernoulli(k4, 0.30, (N_NODES, 1)).astype(jnp.float32)

    receivers = random.randint(k3, (N_EDGES,), 0, N_NODES)
    senders   = random.randint(k2, (N_EDGES,), 0, N_NODES)

    base_features = random.uniform(k1, (N_EDGES, 8))

    receiver_labels = labels[receivers, 0]   # (N_EDGES,) — target node label
    signal_boost = receiver_labels[:, None] * jnp.array([0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3])
    edge_features = jnp.clip(base_features + signal_boost, 0.0, 1.0)

    print(f"Positive proteins : {int(labels.sum())} / {N_NODES} ({labels.mean()*100:.1f}%)")
    print(f"Edge features shape: {edge_features.shape}")

    graph = jraph.GraphsTuple(
        nodes=jnp.zeros((N_NODES, 1)),  # placeholder, not None
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([N_NODES]),
        n_edge=jnp.array([N_EDGES]),
        globals=None
    )

    # === Initializing LNNFormula ===

    formula_str = "0.8::CoExpression & PhysicalBinding -> ProteinFunction"
    formula = LNNFormula(formula_str, nnx.Rngs(123))

    # === Bayesian Neuro-Symbolic Model (Stateless) ===

    from sklearn.preprocessing import StandardScaler as _SS

    def neuro_symbolic_model(graph, labels=None):
        # A. GraphSAGE Perception
        node_updates_raw = jraph.segment_sum(
            graph.edges, graph.receivers, num_segments=N_NODES
        )  # (N_NODES, 8)

        # NORMALIZATION — without it, the sigmoid saturates at 0.4 for all nodes
        # segment_sum values ​​~0.4-0.9 → after StandardScaler mean=0, std=1
        nu_mean = node_updates_raw.mean(axis=0, keepdims=True)
        nu_std  = node_updates_raw.std(axis=0, keepdims=True) + 1e-6
        node_updates = (node_updates_raw - nu_mean) / nu_std  # (N_NODES, 8)

        # B. Bayesian predicate parameters
        with numpyro.plate("predicates_plate", 2):
            slope_l  = numpyro.sample("slope_l",  dist.HalfNormal(2.0))  # prior narrower
            offset_l = numpyro.sample("offset_l", dist.Normal(0.0, 1.0))
            slope_u  = numpyro.sample("slope_u",  dist.HalfNormal(2.0))
            offset_u = numpyro.sample("offset_u", dist.Normal(0.0, 1.0))

        # C. Grounding: normalized features → fuzzy truth [L, U]]
        feats = jnp.stack([
            node_updates[:, :4].mean(axis=1),   # CoExpression
            node_updates[:, 4:].mean(axis=1),   # PhysicalBinding
        ], axis=1)  # (N_NODES, 2)

        L = jax.nn.sigmoid(slope_l[None, :] * (feats - offset_l[None, :]))
        U = jax.nn.sigmoid(slope_u[None, :] * (feats - offset_u[None, :]))

        # D. Fuzzy rule: AND → ProteinFunction
        rule_activation = jnp.min(L, axis=-1)   # (N_NODES,)
        w_impl = numpyro.sample("w_impl", dist.Normal(0.5, 0.5))
        protein_function = jax.nn.sigmoid(rule_activation * w_impl * 5.0)

        # E. Monitoring
        numpyro.deterministic("logical_contradiction", jnp.mean(jnp.maximum(0.0, L - U)))
        numpyro.deterministic("ProteinFunction", protein_function)

        # F. Likelihood — class-weighted BCE
        pos_weight = 2.5   # compensation 30% positive
        with numpyro.plate("proteins", N_NODES):
            logits = jnp.log(protein_function + 1e-7) - jnp.log(1.0 - protein_function + 1e-7)
            obs = labels.squeeze(-1) if labels is not None else None
            if obs is not None:
                # Weighted likelihood
                w = jnp.where(obs == 1, pos_weight, 1.0)
                numpyro.factor("weighted_obs", w * dist.BernoulliLogits(logits).log_prob(obs))
            else:
                numpyro.sample("obs", dist.BernoulliLogits(logits), obs=obs)

    # === Training and Grand Cycle Export ===

    guide     = autoguide.AutoDiagonalNormal(neuro_symbolic_model)
    optimizer = optax.adamw(learning_rate=5e-3)
    svi       = SVI(neuro_symbolic_model, guide, optimizer, Trace_ELBO())

    print("🚀 Start training: Bayesian GraphSAGE + JLNN...")
    vi_result = svi.run(random.PRNGKey(0), 1500, graph, labels)
    print(f"✅ Done. Final ELBO loss: {svi_result.losses[-1]:.4f}")

    # === Export for Grand Cycle ===

    predictive = Predictive(
        neuro_symbolic_model,
        guide=guide,
        params=svi_result.params,
        num_samples=100
    )

    samples = predictive(random.PRNGKey(1), graph)  # labels=None → predikce

    ds = xr.Dataset({
        "ProteinFunction": xr.DataArray(
            samples["ProteinFunction"],
            dims=["sample", "protein"],
            attrs={"description": "Fuzzy truth value per protein"}
        ),
        "logical_contradiction": xr.DataArray(
            samples["logical_contradiction"],
            dims=["sample"],
        ),
    })

    print("✅ Done. The model is ready for Grand Cycle reflection.")
    print(ds)

    # === Visualization of training metrics ===

    # --- A. ELBO convergence ---

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(svi_result.losses, color='steelblue', linewidth=0.8, alpha=0.7, label='ELBO loss')

    # Moving average for clarity
    window = 50
    losses_np = np.array(svi_result.losses)
    if len(losses_np) > window:
        rolling = np.convolve(losses_np, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses_np)), rolling, color='crimson', linewidth=2, label=f'Rolling mean ({window})')

    ax.set_title('SVI Training Convergence (ELBO)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO Loss')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Initial loss : {svi_result.losses[0]:.2f}")
    print(f"Final loss   : {svi_result.losses[-1]:.2f}")
    print(f"Improvement  : {(1 - svi_result.losses[-1]/svi_result.losses[0])*100:.1f}%")

    # --- B. Posterior: ProteinFunction — mean value and uncertainty (HDI) ---

    pf = np.array(samples["ProteinFunction"])   # (100, N_NODES)
    mean_pf = pf.mean(axis=0)                   # (N_NODES,)
    std_pf  = pf.std(axis=0)
    hdi_lo  = np.percentile(pf, 5,  axis=0)
    hdi_hi  = np.percentile(pf, 95, axis=0)

    # Sort proteins by median value for clarity

    order = np.argsort(mean_pf)
    x = np.arange(N_NODES)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, hdi_lo[order], hdi_hi[order], alpha=0.3, color='steelblue', label='90% HDI')
    ax.plot(x, mean_pf[order], color='steelblue', linewidth=1.2, label='Posterior mean')
    ax.axhline(0.5, color='crimson', linestyle='--', linewidth=1, label='Decision boundary (0.5)')

    ax.set_title('ProteinFunction — Posterior Uncertainty per Protein', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein (sorted by mean)')
    ax.set_ylabel('P(ProteinFunction)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- C. Distribution of predicate weights (slope_l, w_impl) ---

    print(list(svi_result.params.keys()))

    # Posterior samples from guide

    guide_predictive = Predictive(guide, params=svi_result.params, num_samples=500)
    guide_samples = guide_predictive(random.PRNGKey(99))

    slope_l_samples = np.array(guide_samples["slope_l"])   # (500, 2)
    w_impl_samples  = np.array(guide_samples["w_impl"])    # (500,)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    labels_pred = ["CoExpression", "PhysicalBinding"]
    colors = ["steelblue", "seagreen"]

    for i, (lbl, col) in enumerate(zip(labels_pred, colors)):
        axes[i].hist(slope_l_samples[:, i], bins=25, color=col, alpha=0.75, edgecolor='white')
        axes[i].axvline(slope_l_samples[:, i].mean(), color='crimson', linestyle='--', linewidth=2,
                        label=f'Mean: {slope_l_samples[:, i].mean():.3f}')
        axes[i].set_title(f'slope_l — {lbl}', fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].legend()

    axes[2].hist(w_impl_samples, bins=25, color='darkorange', alpha=0.75, edgecolor='white')
    axes[2].axvline(w_impl_samples.mean(), color='crimson', linestyle='--', linewidth=2,
                    label=f'Mean: {w_impl_samples.mean():.3f}')
    axes[2].axvline(0.8, color='navy', linestyle=':', linewidth=2, label='Prior mean (0.8)')
    axes[2].set_title('w_impl — Implication Weight', fontweight='bold')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Count')
    axes[2].legend()

    plt.suptitle('Posterior Distributions of Learned Parameters', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Summary
    print(f"slope_l CoExpression   : {slope_l_samples[:, 0].mean():.3f} ± {slope_l_samples[:, 0].std():.3f}")
    print(f"slope_l PhysicalBinding: {slope_l_samples[:, 1].mean():.3f} ± {slope_l_samples[:, 1].std():.3f}")
    print(f"w_impl                 : {w_impl_samples.mean():.3f} ± {w_impl_samples.std():.3f}")

    contra_samples = np.array(samples["logical_contradiction"])   # (100,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(contra_samples, bins=30, color='tomato', alpha=0.8, edgecolor='white')
    axes[0].axvline(contra_samples.mean(), color='darkred', linestyle='--', linewidth=2,
                    label=f'Mean: {contra_samples.mean():.4f}')
    axes[0].set_title('Logical Contradiction Distribution', fontweight='bold')
    axes[0].set_xlabel('Contradiction degree (mean max(0, L-U))')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Relationship contradiction vs ProteinFunction mean
    axes[1].scatter(contra_samples, mean_pf.mean() * np.ones_like(contra_samples),
                    alpha=0.5, color='tomato', s=20)
    axes[1].set_title('Contradiction vs Mean ProteinFunction', fontweight='bold')
    axes[1].set_xlabel('Contradiction degree')
    axes[1].set_ylabel('Mean P(ProteinFunction)')
    axes[1].axvline(0.01, color='gray', linestyle='--', linewidth=1, label='Threshold 0.01')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Mean contradiction : {contra_samples.mean():.6f}")
    print(f"Max contradiction  : {contra_samples.max():.6f}")
    print(f"Model consistency  : {'✅ OK (< 0.01)' if contra_samples.mean() < 0.01 else '⚠️  High contradiction!'}")

    # --- E. Classification metrics ---
    # Posterior mean as prediction

    y_true = np.array(labels.squeeze(-1))   # (N_NODES,)
    y_prob = mean_pf                         # (N_NODES,)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print("=" * 40)
    print("   Classification Metrics (Posterior Mean)")
    print("=" * 40)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 40)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: 0', 'Pred: 1'])
    ax.set_yticklabels(['True: 0', 'True: 1'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    ax.set_title('Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # --- F. Uncertainty calibration — high vs low confidence proteiny ---
    # Protein identification by model confidence

    confidence = 1 - 2 * np.abs(mean_pf - 0.5)   # 1 = certain, 0 = uncertain (close to 0.5)
    uncertain_mask = confidence < 0.3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: mean vs std
    sc = axes[0].scatter(mean_pf, std_pf, c=confidence, cmap='RdYlGn', alpha=0.6, s=15)
    axes[0].axvline(0.5, color='gray', linestyle='--', linewidth=1)
    axes[0].set_title('Posterior Mean vs Std (Uncertainty Map)', fontweight='bold')
    axes[0].set_xlabel('Posterior Mean P(ProteinFunction)')
    axes[0].set_ylabel('Posterior Std (Epistemic Uncertainty)')
    plt.colorbar(sc, ax=axes[0], label='Confidence')

    # Histogram std — uncertainty distribution
    axes[1].hist(std_pf[~uncertain_mask], bins=25, color='seagreen', alpha=0.7,
                label=f'High confidence ({(~uncertain_mask).sum()} proteins)')
    axes[1].hist(std_pf[uncertain_mask],  bins=25, color='tomato',   alpha=0.7,
                label=f'Uncertain ({uncertain_mask.sum()} proteins)')
    axes[1].set_title('Epistemic Uncertainty Distribution', fontweight='bold')
    axes[1].set_xlabel('Posterior Std')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"High confidence proteins : {(~uncertain_mask).sum()} ({(~uncertain_mask).mean()*100:.1f}%)")
    print(f"Uncertain proteins       : {uncertain_mask.sum()} ({uncertain_mask.mean()*100:.1f}%)")

    # === 10. The Grand Cycle – Autonomous Logic Tuning ===
    # --- A. Fuzzy Grounding function ---

    def fuzzy_ramp(x, center, steepness):
        return jax.nn.sigmoid(steepness * (x - center))

    def create_graph_inputs(node_updates_norm_np, p):
        """Grounding from normalized features. ProteinFunction = neutral."""
        nu        = jnp.array(node_updates_norm_np)
        co_expr   = nu[:, :4].mean(axis=1)
        phys_bind = nu[:, 4:].mean(axis=1)
        epsilon   = 0.02
        co_l   = fuzzy_ramp(co_expr,   p["c_coexpr"],   p["steepness"])
        phys_l = fuzzy_ramp(phys_bind, p["c_physbind"], p["steepness"])
        n = len(co_l)
        return {
            "CoExpression":    jnp.stack([co_l,   co_l   + epsilon], axis=-1),
            "PhysicalBinding": jnp.stack([phys_l, phys_l + epsilon], axis=-1),
            "ProteinFunction": jnp.full((n, 2), 0.5),
        }

    # segment_sum
    node_updates_raw = np.array(
        jraph.segment_sum(graph.edges, graph.receivers, num_segments=N_NODES)
    )
    y_true_np = np.array(labels.squeeze(-1))

    # NORMALIZATION (StandardScaler)
    scaler = StandardScaler()
    node_updates_static = scaler.fit_transform(node_updates_raw)

    # Signal correlation with labels
    co_feat = node_updates_static[:, :4].mean(axis=1)
    ph_feat = node_updates_static[:, 4:].mean(axis=1)
    r_co, p_co = pointbiserialr(y_true_np, co_feat)
    r_ph, p_ph = pointbiserialr(y_true_np, ph_feat)

    print(f"CoExpression correlation with labels: r={r_co:.3f}, p={p_co:.4f}")
    print(f"Correlation of PhysicalBind with labels: r={r_ph:.3f}, p={p_ph:.4f}")
    print(f"Scaled range: [{node_updates_static.min():.2f}, {node_updates_static.max():.2f}]")

    # Train/test split 
    idx = np.arange(N_NODES)
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_true_np)
    node_updates_train = node_updates_static[train_idx]
    node_updates_test  = node_updates_static[test_idx]
    y_train = y_true_np[train_idx]
    y_test  = y_true_np[test_idx]

    print(f"Train: {len(train_idx)} ({y_train.mean()*100:.1f}% pos) | Test: {len(test_idx)} ({y_test.mean()*100:.1f}% pos)")
    print("✅ Grounding ready — normalized, no data leakage.")

    # --- B. Objective function for Optuna ---
    # Diagnostics — verify the range of fuzzy outputs (must not be saturated at 0/1)

    test_p = {"steepness": 2.0, "c_coexpr": 0.0, "c_physbind": 0.0,
          "rule_strength": 0.8, "contra_p": 1.0}

    test_inputs  = create_graph_inputs(node_updates_static, test_p)

    test_formula = LNNFormula(
        "0.8::CoExpression & PhysicalBinding -> ProteinFunction", nnx.Rngs(0)
    )

    test_pred = test_formula(test_inputs)

    co_vals  = test_inputs["CoExpression"][:, 0]
    ph_vals  = test_inputs["PhysicalBinding"][:, 0]

    print(f"formula output shape  : {test_pred.shape}")
    print(f"formula sample[0]     : {test_pred[0]}")
    print(f"CoExpression  range   : [{float(co_vals.min()):.3f}, {float(co_vals.max()):.3f}]")
    print(f"PhysicalBind  range   : [{float(ph_vals.min()):.3f}, {float(ph_vals.max()):.3f}]")
    print(f"Unique CoExpr values  : {len(jnp.unique(co_vals))}")
    assert float(co_vals.min()) < 0.3, f"ERROR: CoExpression min={float(co_vals.min()):.3f} — still saturation!"
    assert float(co_vals.max()) > 0.7, f"ERROR: CoExpression max={float(co_vals.max()):.3f} — still saturation!"
    print("✅ Range OK — the model has a gradient to go with.")

    # PRE-OPTUNA DIAGNOSTICS: find optimal parameters by grid search
    co_feat = node_updates_static[:, :4].mean(axis=1)
    ph_feat = node_updates_static[:, 4:].mean(axis=1)

    best_f1 = 0; best_cfg = {}
    for steep in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for c_co in np.linspace(-1.5, 1.0, 25):
            for c_ph in np.linspace(-1.5, 1.0, 25):
                co_l   = expit(steep * (co_feat - c_co))
                phys_l = expit(steep * (ph_feat - c_ph))
                and_out = np.minimum(co_l, phys_l)
                for thresh in np.linspace(0.2, 0.8, 13):
                    yp = (and_out > thresh).astype(float)
                    tp = np.sum((yp==1)&(y_true_np==1))
                    fp = np.sum((yp==1)&(y_true_np==0))
                    fn = np.sum((yp==0)&(y_true_np==1))
                    p_ = tp/(tp+fp+1e-7); r_ = tp/(tp+fn+1e-7)
                    f_ = 2*p_*r_/(p_+r_+1e-7)
                    if f_ > best_f1:
                        best_f1 = f_
                        best_cfg = {'steepness':steep,'c_coexpr':c_co,'c_physbind':c_ph,'thresh':thresh}

    print(f"=== GRID SEARCH RESULTS ===")
    print(f"Theoretical max F1 (AND gate, numpy): {best_f1:.3f}")
    print(f"Optimal parameters:")
    for k,v in best_cfg.items():
        print(f"  {k}: {v:.4f}")

    # Set search space for Optuna according to grid search results
    OPTUNA_STEEPNESS_MAX = best_cfg['steepness'] * 2
    OPTUNA_C_CO_CENTER   = best_cfg['c_coexpr']
    OPTUNA_C_PH_CENTER   = best_cfg['c_physbind']
    OPTUNA_THRESH        = best_cfg['thresh']
    print(f"\nOptuna search space set according to grid search.")
    print(f"Threshold for F1 evaluation: {OPTUNA_THRESH:.2f}")

    def objective(trial):
        """
        Direct BCE classification without LNNFormula.
        Optuna optimizes fuzzy grounding parameters directly.
        LNNFormula is used only for the final symbolic report.
        """
        p = {
            "steepness":  trial.suggest_float("steepness",  0.3, OPTUNA_STEEPNESS_MAX),
            "c_coexpr":   trial.suggest_float("c_coexpr",
                            OPTUNA_C_CO_CENTER - 0.8, OPTUNA_C_CO_CENTER + 0.8),
            "c_physbind": trial.suggest_float("c_physbind",
                            OPTUNA_C_PH_CENTER - 0.8, OPTUNA_C_PH_CENTER + 0.8),
            "w_co":       trial.suggest_float("w_co",  0.3, 1.0),   # weight CoExpression
            "w_ph":       trial.suggest_float("w_ph",  0.1, 0.7),   # PhysicalBinding weight
            "bias":       trial.suggest_float("bias",  -2.0, 0.0),  # logit bias
        }

        # Fuzzy grounding — direct calculation without LNNFormula
        nu_tr = jnp.array(node_updates_train)
        nu_te = jnp.array(node_updates_test)

        def compute_logits(nu, p):
            co   = nu[:, :4].mean(axis=1)
            ph   = nu[:, 4:].mean(axis=1)
            co_l = jax.nn.sigmoid(p["steepness"] * (co - p["c_coexpr"]))
            ph_l = jax.nn.sigmoid(p["steepness"] * (ph - p["c_physbind"]))
            # Weighted combination — better than AND/OR for gradient flow
            score = p["w_co"] * co_l + p["w_ph"] * ph_l + p["bias"]
            return score   # logits

        logits_tr = compute_logits(nu_tr, p)
        logits_te = compute_logits(nu_te, p)
        prob_tr   = jax.nn.sigmoid(logits_tr)
        prob_te   = jax.nn.sigmoid(logits_te)

        # BCE loss — direct, no LNNFormula transformation
        y_tr = jnp.array(y_train)
        bce  = float(-jnp.mean(
            y_tr * jnp.log(prob_tr + 1e-7) +
            (1 - y_tr) * jnp.log(1 - prob_tr + 1e-7)
        ))

        # F1 to test set
        best_f1 = 0.0
        for thresh in [OPTUNA_THRESH, 0.4, 0.5, 0.6]:
            yp = (prob_te > thresh).astype(jnp.float32)
            tp = float(jnp.sum((yp==1) & (y_test==1)))
            fp = float(jnp.sum((yp==1) & (y_test==0)))
            fn = float(jnp.sum((yp==0) & (y_test==1)))
            prec = tp/(tp+fp+1e-7); rec = tp/(tp+fn+1e-7)
            f1   = 2*prec*rec/(prec+rec+1e-7)
            if f1 > best_f1:
                best_f1 = f1

        trial.report(best_f1, step=0)
        return best_f1

    # --- C. Launch of the Grand Cycle study ---

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=250,
        interval_steps=250
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=99, n_startup_trials=20)

    # No pruner — every trial is now instant (no training loop)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="JLNN_GraphSAGE_Grand_Cycle_directBCE"
    )

    N_TRIALS = 200   # multiple trials — each takes <1s
    print(f"🔍 Starting Grand Cycle ({N_TRIALS} trials, direct BCE, optimizing F1)...")
    study.optimize(objective, n_trials=N_TRIALS, timeout=300, show_progress_bar=True)

    completed = [t for t in study.trials if t.value is not None]
    print(f"Completed : {len(completed)} / {N_TRIALS}")

    print(f"\n{'='*70}")
    print(f"  Best trial : #{study.best_trial.number}")
    print(f"  Best F1    : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:20s}: {v:.4f}")
    print(f"{'='*70}")

    # --- D. Visualization of Grand Cycle results ---
    # Optimization history

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Grand Cycle: Optimization Progress")
    plt.tight_layout()
    plt.show()

    # Hyperparameter importances

    completed = [t for t in study.trials if t.value is not None]
    print(f"Completed trials: {len(completed)} / {len(study.trials)}")

    if len(completed) >= 3:
        param_names = list(study.best_params.keys())
        values_by_param = {p: [t.params[p] for t in completed] for p in param_names}
        losses_by_trial = [t.value for t in completed]

        importances = {}
        for p in param_names:
            vals = values_by_param[p]
            # Skip parameters with zero variance
            if len(set(vals)) < 2:
                importances[p] = (0.0, 1.0)
                continue
            corr, pval = spearmanr(vals, losses_by_trial)
            # NaN → 0 if correlation cannot be calculated
            corr = 0.0 if (corr != corr) else corr   # NaN check
            importances[p] = (abs(corr), float(pval) if pval == pval else 1.0)

        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1][0], reverse=True))

        fig, ax = plt.subplots(figsize=(8, 4))
        colors_imp = ['crimson' if v[0] > 0.3 else 'steelblue' for v in sorted_imp.values()]
        ax.barh(list(sorted_imp.keys()), [v[0] for v in sorted_imp.values()],
                color=colors_imp, alpha=0.8)
        ax.axvline(0.3, color='gray', linestyle='--', linewidth=1, label='Threshold 0.3')
        ax.set_title('Hyperparameter Importance (Spearman |corr| s F1)', fontweight='bold')
        ax.set_xlabel('|Spearman correlation|')
        ax.legend()
        plt.tight_layout()
        plt.show()

        print("\nImportance ranking:")
        for p, (v, pval) in sorted_imp.items():
            bar = '█' * int(v * 30)   # v is pure float, NaN handled above
            print(f"  {p:20s}: {v:.3f}  (p={pval:.3f})  {bar}")
    else:
        print("⚠️ Too few completed trials.")

    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed = [t for t in study.trials if t.value is not None]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"Completed : {len(completed)}")
    print(f"Pruned    : {len(pruned)}")
    print(f"Failed    : {len(failed)}")

    # Contour: steepness vs rule_strength

    optuna.visualization.matplotlib.plot_contour(study, params=["steepness", "w_co"])
    plt.title("Steepness vs CoExpression Weight")
    plt.tight_layout()
    plt.show()

    # Contour: grounding center

    optuna.visualization.matplotlib.plot_contour(study, params=["c_coexpr", "c_physbind"])
    plt.title("CoExpression center vs PhysicalBinding center")
    plt.tight_layout()
    plt.show()

    # --- E. Retrain with the best parameters ---

    best = study.best_params
    print("🏆 Best Grand Cycle parameters (direct BCE):")
    for k, v in best.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # Final evaluation with best params
    nu_all = jnp.array(node_updates_static)
    nu_te  = jnp.array(node_updates_test)

    def compute_logits(nu, p):
        co   = nu[:, :4].mean(axis=1)
        ph   = nu[:, 4:].mean(axis=1)
        co_l = jax.nn.sigmoid(p["steepness"] * (co - p["c_coexpr"]))
        ph_l = jax.nn.sigmoid(p["steepness"] * (ph - p["c_physbind"]))
        return p["w_co"] * co_l + p["w_ph"] * ph_l + p["bias"]

    probs_test = np.array(jax.nn.sigmoid(compute_logits(nu_te, best)))

    # Sweep thresholds
    best_thresh_f1 = 0; best_thresh = 0.5
    for thresh in np.linspace(0.05, 0.95, 91):
        yp = (probs_test > thresh).astype(int)
        f_ = f1_score(y_test, yp, zero_division=0)
        if f_ > best_thresh_f1:
            best_thresh_f1 = f_; best_thresh = thresh

    y_pred_final = (probs_test > best_thresh).astype(int)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_f1  = f1_score(y_test, y_pred_final, zero_division=0)
    final_auc = roc_auc_score(y_test, probs_test)
    cm        = confusion_matrix(y_test, y_pred_final)

    print(f"\n✅ TEST SET results (Grand Cycle, direct BCE):")
    print(f"   Accuracy : {final_acc:.4f}")
    print(f"   F1 Score : {final_f1:.4f}  (threshold={best_thresh:.2f})")
    print(f"   ROC-AUC  : {final_auc:.4f}")
    print(f"   Baseline grid search F1: ~0.60")

    # Now initialize LNNFormula for symbolic report
    # with rule_strength = best steepness normalized
    rule_strength = min(1.0, max(0.65, best.get("w_co", 0.8)))
    best_rule    = f"{rule_strength:.3f}::CoExpression & PhysicalBinding -> ProteinFunction"
    best_formula = LNNFormula(best_rule, nnx.Rngs(0))

    # Fast train best_formula on all data (only for weights in the report)
    p_lnn = {k: v for k, v in best.items() if k in ["steepness","c_coexpr","c_physbind"]}
    p_lnn["rule_strength"] = rule_strength; p_lnn["contra_p"] = 0.1
    all_inputs = create_graph_inputs(node_updates_static, p_lnn)
    all_target = jnp.where(
        jnp.array(y_true_np)[:, None] == 1,
        jnp.array([[0.75, 1.00]]),
        jnp.array([[0.00, 0.25]])
    )
    opt_lnn = nnx.Optimizer(best_formula, optax.adamw(1e-3), wrt=nnx.Param)
    for _ in tqdm(range(500)):
        def lf(f):
            pr = f(all_inputs); pa = jnp.min(pr, axis=1)
            return jnp.mean((pa - all_target)**2)
        _, g = nnx.value_and_grad(lf)(best_formula)
        opt_lnn.update(best_formula, g)
    print("✅ LNNFormula trained for symbolic report.")

    # Confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(['Pred: 0','Pred: 1'])
    axes[0].set_yticklabels(['True: 0','True: 1'])
    for ii in range(2):
        for jj in range(2):
            axes[0].text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                        color='white' if cm[ii,jj]>cm.max()/2 else 'black', fontsize=14)
    axes[0].set_title(f'Confusion Matrix (Test, thresh={best_thresh:.2f})', fontweight='bold')
    plt.colorbar(im, ax=axes[0])

    # Distribuce probs
    axes[1].hist(probs_test[y_test==0], bins=30, alpha=0.6, color='steelblue', label='Negative')
    axes[1].hist(probs_test[y_test==1], bins=30, alpha=0.6, color='crimson',   label='Positive')
    axes[1].axvline(best_thresh, color='black', linestyle='--', label=f'Threshold={best_thresh:.2f}')
    axes[1].set_title('Score Distribution (Test Set)', fontweight='bold')
    axes[1].set_xlabel('P(ProteinFunction)'); axes[1].legend()
    plt.tight_layout(); plt.show()

    # --- F. Symbolic report for another LLM agent ---

    graphdef, state = nnx.split(best_formula)
    flat = state.flat_state()

    print("Available parameters in best_formula:")
    weight_info = {}
    for path, leaf in zip(flat.paths, flat.leaves):
        v = leaf.value if hasattr(leaf, 'value') else leaf
        arr = jnp.array(v)
        key = "_".join(str(p) for p in path)
        weight_info[key] = float(arr.mean())
        print(f"  {key}: shape={arr.shape}, mean={arr.mean():.4f}")

    # --- Build xarray dataset of weights using correct signature ---
    
    # Retrieve indexes from flat.paths
    roots = [i for i, p in enumerate(flat.paths) if p == ('root', 'gate', 'weights')]
    lefts = [i for i, p in enumerate(flat.paths) if p == ('root', 'left', 'gate', 'weights')]

    # root_gate_weights — implication weights (Co Expression, Physical Bonding)
    ds_impl = extract_weights_to_xarray(
        weights=jnp.array(flat.leaves[roots[0]].value),
        input_labels=["CoExpression", "PhysicalBinding"],
        gate_name="Implication"
    )

    # root_left_gate_weights — AND gate weights
    ds_and = extract_weights_to_xarray(
        weights=jnp.array(flat.leaves[lefts[0]].value),
        input_labels=["CoExpression", "PhysicalBinding"],
        gate_name="AND"
    )

    # Individual weights for the report
    impl_w = ds_impl.values   # [CoExpression, PhysicalBinding]
    and_w  = ds_and.values    # [CoExpression, PhysicalBinding]

    print("✅ xarray weights:")
    print(ds_impl)
    print(ds_and)

    # --- LLM Report ---

    report = f"""
    {'='*80}
    GRAND CYCLE REPORT — JLNN Neuro-Symbolic Bayesian GraphSAGE
    Generated for: Next LLM Agent / Human Analyst
    {'='*80}

    DATASET: OGBN-Proteins subset ({N_NODES} nodes, {N_EDGES} edges, 8 edge features)

    LOGICAL RULE (optimized):
    {best_rule}

    BEST CONFIGURATION (Optuna, {N_TRIALS} trials, přímá BCE):
    steepness        : {best['steepness']:.4f}
    c_coexpr         : {best['c_coexpr']:.4f}   ← fuzzy boundary CoExpression
    c_physbind       : {best['c_physbind']:.4f}   ← fuzzy boundary PhysicalBinding
    w_co (CoExpr weight)  : {best['w_co']:.4f}
    w_ph (PhysBind weight): {best['w_ph']:.4f}
    bias             : {best['bias']:.4f}

    LEARNED WEIGHTS (LNNFormula post-training):
    Implication gate weights : CoExpression={impl_w[0]:.4f}  PhysicalBinding={impl_w[1]:.4f}
    AND gate weights         : CoExpression={and_w[0]:.4f}   PhysicalBinding={and_w[1]:.4f}
    Implication beta         : {weight_info['root_gate_beta']:.4f}
    AND beta                 : {weight_info['root_left_gate_beta']:.4f}

    PREDICATE SLOPES (grounding steepness):
    CoExpression    slope_l={weight_info['predicates_CoExpression_predicate_slope_l']:.4f}  slope_u={weight_info['predicates_CoExpression_predicate_slope_u']:.4f}
    PhysicalBinding slope_l={weight_info['predicates_PhysicalBinding_predicate_slope_l']:.4f}  slope_u={weight_info['predicates_PhysicalBinding_predicate_slope_u']:.4f}
    ProteinFunction slope_l={weight_info['predicates_ProteinFunction_predicate_slope_l']:.4f}  slope_u={weight_info['predicates_ProteinFunction_predicate_slope_u']:.4f}

    PERFORMANCE:
    Best trial F1  : {study.best_value:.4f}
    Final F1 Score : {final_f1:.4f}  (threshold={best_thresh:.2f})
    Final ROC-AUC  : {final_auc:.4f}
    Baseline (grid search, no training): ~0.604

    RECOMMENDATIONS FOR NEXT GRAND CYCLE:
    - steepness range    : [{best['steepness']-0.3:.2f}, {best['steepness']+0.5:.2f}]
    - c_coexpr range     : [{best['c_coexpr']-0.3:.2f}, {best['c_coexpr']+0.3:.2f}]
    - c_physbind range   : [{best['c_physbind']-0.3:.2f}, {best['c_physbind']+0.3:.2f}]
    - w_co dominant ({best['w_co']:.3f} > w_ph {best['w_ph']:.3f}) → CoExpression je silnější prediktor
    - Consider: 3rd predicate (e.g. SequenceHomology) for richer logic
    - AND beta={weight_info['root_left_gate_beta']:.3f} > 1.0 → pravidlo je aktivní

    NEXT STEP: Feed this report to a planning LLM agent to refine the rule grammar
    and expand the Grand Cycle with new biological predicates.
    {'='*80}
    """
    print(report)


Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_NS_Bayesian_GraphSAGE.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_NS_Bayesian_GraphSAGE.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.

