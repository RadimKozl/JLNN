Weighted Rules & Multiple Antecedents
======================================

This demo focuses on advanced modeling of logical relationships, where not only the input data plays a role, but also its relative importance and the overall plausibility of the rule.
We work with so-called "Network Surgery". We show how to programmatically access the internal parameters (weights) of an already compiled JLNN logic graph, which is crucial for learning or expert tuning of models.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_weighted_rules.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_weighted_rules.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.

Content Overview
-----------------

In this section, we define a rule with a fixed weight of 0.8 and then change the weight of the second input (*B*) in a loop to observe how the uncertainty of the result changes.

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
    
    import jax.numpy as jnp
    from flax import nnx
    import matplotlib.pyplot as plt

    # JLNN core imports
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.nn.predicates import FixedPredicate
    from jlnn.nn.gates import WeightedAnd, WeightedImplication

    print("JLNN framework ready.")

    base_rule = "0.8 :: A & B -> C"

    crisp_inputs = {
        "A": jnp.array([[1.0, 1.0]]),
        "B": jnp.array([[1.0, 1.0]]),
        "C": jnp.array([[0.0, 1.0]])    # The goal begins as complete uncertainty
    }

    fuzzy_inputs = {
        "A": jnp.array([[0.80, 0.95]]),
        "B": jnp.array([[0.60, 0.85]]),
        "C": jnp.array([[0.0, 1.0]])    # The goal begins as complete uncertainty
    }

    def run_weighted_rule(w: float, inputs: dict):
        # 1. Compiling the rule
        model = LNNFormula(base_rule, nnx.Rngs(42))

        # 2. Scales Adjustment (Surgery)
        for path, module in nnx.iter_modules(model):
            if isinstance(module, WeightedAnd):
                module.weights.value = jnp.array([1.0, w])
                if hasattr(module, 'beta'):
                    module.beta.value = jnp.array(1.0)

            if isinstance(module, WeightedImplication):
                if hasattr(module, 'beta'):
                    module.beta.value = jnp.array(1.0)

        # 3. Grounding: Switch to FixedPredicate for all nodes in inputs
        for name in inputs:
            if name in model.predicates:
                model.predicates[name].predicate = FixedPredicate()

        # 4. Inference
        # Now KeyError won't occur because "C" exists in inputs
        output = model(inputs)

        # 5. Extracting the result [L, U]
        flat_output = jnp.reshape(output, (-1, 2))
        L, U = flat_output[0, 0].item(), flat_output[0, 1].item()
        width = U - L

        return L, U, width
    
    weights = jnp.linspace(0.0, 1.0, 11)

    crisp_results = []
    fuzzy_results = []

    print(f"{'Váha w':<8} | {'Crisp C [L, U]':<18} | {'Fuzzy C [L, U]':<18}")
    print("-" * 55)

    for w in weights:
        cL, cU, cW = run_weighted_rule(float(w), crisp_inputs)
        fL, fU, fW = run_weighted_rule(float(w), fuzzy_inputs)

        crisp_results.append((cL, cU, cW))
        fuzzy_results.append((fL, fU, fW))

        print(f"{float(w):<8.1f} | [{cL:.3f}, {cU:.3f}] | [{fL:.3f}, {fU:.3f}]")

    c_L, c_U, c_width = zip(*crisp_results)
    f_L, f_U, f_width = zip(*fuzzy_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Graf 1: Intervaly ---
    ax1.fill_between(weights, c_L, c_U, color='blue', alpha=0.1, label='Crisp Range')
    ax1.plot(weights, c_L, '--', color='blue', alpha=0.6, label='Crisp Lower (L)')
    ax1.plot(weights, c_U, '-', color='blue', label='Crisp Upper (U)')
    ax1.fill_between(weights, f_L, f_U, color='orange', alpha=0.1, label='Fuzzy Range')
    ax1.plot(weights, f_L, '--', color='orange', alpha=0.6, label='Fuzzy Lower (L)')
    ax1.plot(weights, f_U, '-', color='orange', label='Fuzzy Upper (U)')
    ax1.axhline(y=0.8, color='red', linestyle=':', alpha=0.5, label='Rule Weight (0.8)')
    ax1.set_title("Influence of Antecedent Weight on Output C")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower left', fontsize=9, ncol=2)
    ax1.set_xlabel('Weight on antecedent B (w)')
    ax1.set_ylabel("Interval Width")
    ax1.grid(True, alpha=0.2)

    # --- Chart 2: Width of uncertainty ---
    # We use linewidth=3 and zorder to make the lines "break" the axis
    ax2.plot(weights, c_width, 'o-', label='Crisp Uncertainty', color='blue',
            linewidth=3, markersize=7, zorder=5)
    ax2.plot(weights, f_width, 's-', label='Fuzzy Uncertainty', color='orange',
            linewidth=3, markersize=7, zorder=4)

    # Input width reference line B (0.25)
    b_uncertainty = 0.85 - 0.60
    ax2.axhline(y=b_uncertainty, color='black', linestyle='--', alpha=0.3,
                label=f'Input B Uncertainty ({b_uncertainty:.2f})')

    ax2.set_title("Uncertainty Propagation (U - L)")
    ax2.set_xlabel("Weight on antecedent B (w)")
    ax2.set_ylabel("Interval Width")


    # THIS LINE IS KEY:
    # We set the upper limit according to the real data (fuzzy_width),
    # so that the graph is not "drowned" in the 0-1 scale
    max_w = max(max(f_width), b_uncertainty) * 1.2
    ax2.set_ylim(-0.02, max_w if max_w > 0 else 0.4)

    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.figtext(0.5, 0.01, "At w=0.0: Output depends on A. | At w=1.0: Full influence of B.",
                ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_weighted_rules.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_weighted_rules.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.