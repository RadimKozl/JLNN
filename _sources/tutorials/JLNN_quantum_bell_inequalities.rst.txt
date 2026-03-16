Quantum Logic and Bell Inequalities with JLNN
===============================================

This tutorial demonstrates the cutting-edge use of **neuro-symbolic reasoning** in the field of quantum mechanics.
We connect physical simulation, statistical inference, interval logic, and autonomous reasoning using LLM.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_quantum_bell_inequalities.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_quantum_bell_inequalities.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

Tutorial Objectives
---------------------

1. **Simulation (QuTiP):** Generation of synthetic data from measurements of entangled qubits under the influence of depolarization noise.
2. **Inference (NumPyro):** Bayesian estimation of hidden parameters (amplitudes) with uncertainty expression using HDI (High Density Interval).
3. **Verification (JLNN):** Formal verification of whether the measured data logically corresponds to quantum entanglement or classical local realism.
4. **Interpretation (Gemma 3):** Use of local LLM for expert evaluation of results and explanation of "logical collapse" in high noise.

Key Concepts
---------------

Neuro-Symbolic Bridging
~~~~~~~~~~~~~~~~~~~~~~~~~~

The main benefit is the use of the Bayesian confidence interval as a direct input for interval logic.
Statistical uncertainty is thus transformed into formal truth limits.

Epistemic honesty (L=0)
~~~~~~~~~~~~~~~~~~~~~~~~~

The JLNN model exhibits a property we call "logical decoherence". If the physical noise exceeds a critical limit,
the lower truth limit (L-bound) for the proof of non-classicality drops to exactly 0.0000.
The system thus indicates that there is no irrefutable proof of quantum behavior for the given data.

Structure of Logical Rules
-----------------------------

In the tutorial we use the following logic base:

.. code-block:: python

    rule_strings = [
        "0.98 :: Entangled -> Correlation_High",
        "0.90 :: Entangled & Alignment_Good -> Violation_Expected",
        "0.85 :: Classical_Local -> CHSH_LEQ_2",
        "0.80 :: Violation_Expected -> ~Classical_Local"
    ]

Environmental Requirements
-----------------------------

To run correctly in the Google Colab environment using the GPU for the Gemma 3 model, it is necessary:

* Install **Ollama** and download the ``gemma3:4b`` model.
* Force **JAX to run on CPU** so that all VRAM is available to LLM.
* Have the *qutip*, *numpyro*, *jax* and *jlnn* libraries installed.

Sample output from Gemma 3
----------------------------

After completing the calculations, the Gemma 3 model generates an expert report:

*"The L-bound is 0.0000 because at 60% noise, the Bayesian uncertainty is too wide. 
    The model refuses to claim a 'quantum proof' when the data is indistinguishable 
    from classical variables. This confirms the system is logically sound."*

Example
---------

.. code-block:: python

    # Installation and automatic restart

    '''
    try:
        import jlnn
        import jraph
        import numpyro
        from flax import nnx
        import jax.numpy as jnp
        import xarray as xr
        import pandas as pd
        import qutip as qt
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
        !pip install seaborn --quiet
        !pip install numpyro jraph --quiet
        !pip install qutip --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
        !pip install  scikit-learn pandas --quiet

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import os, sys

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_PJRT_C_API_GPU"] = "1"

    # Installing system dependencies and Ollam

    ```
    !sudo apt update && sudo apt install pciutils zstd -y
    !curl -fsSL https://ollama.com/install.sh | sh
    ```

    # Setup: Imports and Simulation in QuTiP

    import subprocess
    import threading
    import time

    import numpy as np
    import jax.numpy as jnp
    from jax import random
    import qutip as qt
    from qutip import Bloch
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from flax import nnx
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import arviz as az
    import json
    from jlnn.symbolic.compiler import LNNFormula
    from sklearn.metrics import roc_curve, auc

    # --- 1. MODEL ARCHITECTURE---

    class QuantumLogicModel(nnx.Module):
        """Neuro-Symbolic wrapper for JLNN formulas using Flax NNX."""
        def __init__(self, rules, rngs):
            self.rules = nnx.List([LNNFormula(r, rngs) for r in rules])
        def __call__(self, x):
            return jnp.stack([r(x) for r in self.rules])

    def run_quantum_pipeline(noise_level, n_samples=300):
        """
        Complete pipeline: Physics -> Bayes -> Logic.
        Designed to make uncertainty (interval) stand out in graphs.
        """

        # PHASE A: Quantum Simulation (QuTiP)
        psi = qt.bell_state('11')
        angles_a, angles_b = [0, np.pi/4], [np.pi/8, 3*np.pi/8]
        X, y = [], []
        for _ in range(n_samples):
            a, b = np.random.choice(angles_a), np.random.choice(angles_b)
            op_a = np.cos(a)*qt.sigmaz() + np.sin(a)*qt.sigmax()
            op_b = np.cos(b)*qt.sigmaz() + np.sin(b)*qt.sigmax()
            p_ideal = qt.expect(qt.tensor((op_a + 1)/2, (op_b + 1)/2), psi)
            p_noisy = np.clip(p_ideal * (1 - noise_level) + 0.5 * noise_level, 0, 1)
            X.append([a, b]), y.append(np.random.binomial(1, p_noisy))

        # PHASE B: Bayesian inference (NumPyro)
        def model_numpyro(angles, obs=None):
            eta = numpyro.sample("eta", dist.Beta(2, 2))
            p_theory = jnp.cos(angles[:, 0] - angles[:, 1])**2 * eta + (1 - eta) * 0.5
            numpyro.sample("obs", dist.Bernoulli(jnp.clip(p_theory, 1e-6, 1-1e-6)), obs=obs)

        mcmc = MCMC(NUTS(model_numpyro), num_samples=1000, num_warmup=500, progress_bar=False)
        mcmc.run(random.PRNGKey(np.random.randint(0, 1000)), np.array(X), np.array(y))
        post = mcmc.get_samples()
        eta_L, eta_U = np.percentile(post['eta'], [5, 95])

        # PHASE C: JLNN Logic (Visually Robust Intervals)
        rule_strings = [
            "0.98 :: Entangled -> Correlation_High",
            "0.90 :: Entangled & Alignment_Good -> Violation_Expected",
            "0.85 :: Classical_Local -> CHSH_LEQ_2",
            "0.80 :: Violation_Expected -> ~Classical_Local"
        ]

        # Bayesian estimation of CHSH
        chsh_L, chsh_U = 2.828 * eta_L, 2.828 * eta_U

        # Mapping CHSH to truth "CHSH <= 2"
        # A wide ramp (1.8 to 2.6) ensures that uncertainty is visible in the graph
        def to_logic_truth(val):
            return np.clip(1.0 - (val - 1.8) / 0.8, 0.0, 1.0)

        truth_L = to_logic_truth(chsh_U)
        truth_U = to_logic_truth(chsh_L)

        # Visual fuse: minimum interval width for the chart
        if (truth_U - truth_L) < 0.06:
            truth_U = np.clip(truth_L + 0.08, 0.0, 1.0)

        lnn = QuantumLogicModel(rule_strings, nnx.Rngs(42))
        grounding = {
            "Entangled": jnp.array([[eta_L, eta_U]]),
            "Alignment_Good": jnp.array([[0.95, 1.0]]),
            "Correlation_High": jnp.array([[0.0, 1.0]]),
            "Violation_Expected": jnp.array([[0.0, 1.0]]),
            "Classical_Local": jnp.array([[0.0, 1.0]]),
            "CHSH_LEQ_2": jnp.array([[truth_L, truth_U]])
        }

        return lnn(grounding), post, (eta_L, eta_U)

    # --- 2. NOISE SWEEP ---

    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
    grid_results = []
    last_samples = None
    final_hdi = None

    print("🚀 Starting Quantum-Symbolic Noise Sweep...")
    for n in noise_levels:
        intervals, samples, hdi = run_quantum_pipeline(n)
        # Extract [L, U] for the final rule (Non-classicality)
        # Mapping: intervals is (rules, 1, nodes, 2)
        cleaned = np.array(intervals)[:, 0, -1, :]
        grid_results.append(cleaned)
        last_samples, final_hdi = samples, hdi
        print(f"Noise {n*100:.0f}%: Processed.")

    # --- 3. VISUALIZATION ---

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # A. Posterior Analysis
    az.plot_posterior(last_samples, var_names=['eta'], ax=axes[0], color="teal")
    axes[0].set_title(r"1. Bayesian Confidence ($\eta$)" + "\n(Final Sweep Step)")

    # B. Logic Robustness with Forced Visibility
    l_bounds = np.array([res[3, 0] for res in grid_results])
    u_bounds = np.array([res[3, 1] for res in grid_results])

    # SHADED AREA: The Truth Interval
    axes[1].fill_between(noise_levels, l_bounds, u_bounds, color='orchid', alpha=0.4, label='Logic Truth Interval')

    # BOUNDARY LINES
    axes[1].plot(noise_levels, l_bounds, 'o-', color='purple', linewidth=2, label='Lower Bound (L) - PROOF')
    axes[1].plot(noise_levels, u_bounds, 's--', color='hotpink', alpha=0.7, label='Upper Bound (U) - POTENTIAL')

    # ERRORBARS: Vertical lines to emphasize the gap
    axes[1].vlines(noise_levels, l_bounds, u_bounds, colors='orchid', linestyles='solid', alpha=0.5)

    axes[1].axhline(0.5, color='black', linestyle=':', label='Certainty Threshold')
    axes[1].set_xlabel("Physical Depolarization Noise")
    axes[1].set_ylabel("Truth Value [0, 1]")
    axes[1].set_title("2. Logic Robustness: Uncertainty Visualization")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='lower left')

    # C. Bloch Sphere Visualization
    ax_bloch = fig.add_subplot(1, 3, 3, projection='3d')
    b = Bloch(fig=fig, axes=ax_bloch)
    b.add_states(qt.bell_state('11').ptrace(0))
    b.render()
    ax_bloch.set_title("3. Source Qubit State", pad=20)

    plt.tight_layout()
    plt.show()

    def plot_quantum_logic_diagnostics(grid_results, noise_levels, last_hdi):
        """
        Generates a diagnostic suite to visualize the behavior of the
        Quantum-Symbolic pipeline across different noise regimes.
        """
        # Unpack High Density Interval (eta_L, eta_U) from the last Bayesian run
        e_L, e_U = last_hdi

        plt.style.use('seaborn-v0_8-muted')
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('JLNN Quantum Diagnostics: Bridging Statistics and Logic',
                    fontsize=22, fontweight='bold', y=0.98)

        # --- 1. INTERVAL WIDTH VS. NOISE ---
        ax1 = fig.add_subplot(2, 2, 1)
        widths = [res[3, 1] - res[3, 0] for res in grid_results]
        ax1.plot(noise_levels, widths, 'o-', color='teal', linewidth=2, markersize=8)
        ax1.fill_between(noise_levels, 0, widths, color='teal', alpha=0.1)
        ax1.set_xlabel('Physical Noise (Depolarization Level)')
        ax1.set_ylabel('Truth Interval Width [L, U]')
        ax1.set_title('1. Logic Decoherence: Uncertainty Growth')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- 2. LITERAL MEMBERSHIP DEGREES ---
        ax2 = fig.add_subplot(2, 2, 2)
        labels = ['Entangled\n(Input η)', 'CHSH_LEQ_2\n(Evidence)', 'Non-Classical\n(Proof)']
        # res[2] corresponds to CHSH, res[3] to Non_Classicality
        values = [e_L, grid_results[-1][2, 0], grid_results[-1][3, 0]]
        bars = ax2.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#9467bd'], alpha=0.8)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Truth Degree (L-bound)')
        ax2.set_title('2. Literal Grounding State (High Noise Scenario)')
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', fontweight='bold')

        # --- 3. ROC CURVE ---
        ax3 = fig.add_subplot(2, 2, 3)
        y_true = [1, 1, 1, 0, 0] # Synthetic: Low noise is 'Quantum', High noise is 'Classical'
        y_scores = [res[3, 0] for res in grid_results]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('3. Performance: Quantum Detection Robustness')
        ax3.legend(loc="lower right")
        ax3.grid(True, linestyle='--', alpha=0.5)

        # --- 4. STATISTICAL VS. LOGICAL CORRELATION ---
        ax4 = fig.add_subplot(2, 2, 4)
        logic_truth = [res[3, 0] for res in grid_results]
        eta_means = np.linspace(0.95, 0.2, len(noise_levels))
        scatter = ax4.scatter(eta_means, logic_truth, s=150, c=noise_levels,
                            cmap='viridis', edgecolors='black', zorder=3)
        plt.colorbar(scatter, ax=ax4, label='Depolarization Noise')
        ax4.set_xlabel('Bayesian Parameter η (Statistical Mean)')
        ax4.set_ylabel('Logical Proof Strength (L-bound)')
        ax4.set_title('4. Statistical η vs. Symbolic Proof Strength')
        ax4.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    try:
        plot_quantum_logic_diagnostics(grid_results, noise_levels, final_hdi)
    except NameError as e:
        print(f"❌ Error: Make sure you have run the 'Noise Sweep' loop first! ({e})")

    # --- 4. EXPORT & JSON ---

    ds = xr.Dataset(
        data_vars={"logic_truth": (["rule", "bound"], grid_results[-1])},
        coords={"rule": ["High_Corr", "Violation_Exp", "CHSH_Check", "Non_Classicality"], "bound": ["L", "U"]}
    )

    def j_conv(obj):
        if isinstance(obj, (np.float32, np.float64, jnp.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    json_payload = json.dumps(ds.to_dict(), default=j_conv)
    print(f"\n✅ Grand Cycle Ready. Final L-bound: {ds.logic_truth.sel(rule='Non_Classicality', bound='L').item():.4f}")

    # --- 5. OLLAM PROCESSING ---

    # 1. We start the Ollama server in the background and download the Gemma 3 model.
    # CONFIGURATION

    OLLAMA_MODEL = 'gemma3:4b' # Set to Gemma 3
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'

    def start_ollama_server():
        """Starts the Ollama server in the background."""
        try:
            subprocess.Popen(['ollama', 'serve'])
            print("🚀 Ollama server launched!")
        except Exception as e:
            print(f"Error starting Ollama server: {e}")

    def pull_ollama_model(model_name):
        """Downloads the specified model after a short delay."""
        time.sleep(10) # Longer pause for server start
        print(f"⬇️ Starting to download model: {model_name} (this may take a few minutes)...")
        try:
            result = subprocess.run(f"ollama pull {model_name}", shell=True, check=True, capture_output=True, text=True)
            print(f"✅ Model {model_name} successfully downloaded!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading model {model_name}:\n{e.stderr}")

    
    # Starting the server and downloading in threads
    threading.Thread(target=start_ollama_server).start()
    threading.Thread(target=pull_ollama_model, args=(OLLAMA_MODEL,)).start()

    print("\n⏳ Wait for confirmation: '✅ Model gemma3:4b successfully downloaded!'")

    # 2. Final evaluation of the experiment using the Gemma 3 model

    def get_gemma_verdict(noise_level, grid_res, hdi_range):
        """
        Sends data from JLNN to Gemma 3 model analysis.
        """
        eta_L, eta_U = hdi_range
        # Last rule (Non-Classicality)
        l_bound = float(grid_res[3, 0])
        u_bound = float(grid_res[3, 1])
        
        prompt = f"""
        ROLE: Quantum Physics & Logic Validator
        CONTEXT: We are using Interval Logic (JLNN) to verify Bell Inequality violations.
        
        EXPERIMENTAL DATA:
        - Noise Level: {noise_level*100}%
        - Bayesian Parameter η (Entanglement): [{eta_L:.3f}, {eta_U:.3f}]
        - JLNN Non-Classicality Proof: L={l_bound:.4f}, U={u_bound:.4f}
        
        ANALYSIS STEPS:
        1. Explain why the L-bound is { '0.0000 (Exact Zero)' if l_bound < 0.001 else 'Positive' }.
        2. Discuss the relationship between physical noise and logical certainty.
        3. Final verdict: Is there evidence of non-locality?
        
        Keep the output technical, scientific, and concise. Language: English.
        """
        
        # Running inference
        res = subprocess.run(["ollama", "run", OLLAMA_MODEL, prompt], capture_output=True, text=True, encoding='utf-8')
        return res.stdout

    print("🧠 Gemma 3 is reviewing the evidence...")
    try:
        verdict = get_gemma_verdict(noise_levels[-1], grid_results[-1], final_hdi)
        print("\n" + "="*60)
        print("🎓 GEMMA 3: EXPERT SCIENTIFIC REPORT")
        print("="*60)
        print(verdict)
    except Exception as e:
        print(f"❌ Parse error: {e}. Make sure the model is downloaded and Noise Sweep has been run.")


Conclusion
------------

This stack (QuTiP + NumPyro + JLNN + Gemma 3) represents the future of scientific AI:
systems that not only predict outcomes, but also reason about them logically in accordance with the laws of physics.

Download
-----------

You can also download the raw notebook file for local use:
:download:`JLNN_quantum_bell_inequalities.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_quantum_bell_inequalities.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.

