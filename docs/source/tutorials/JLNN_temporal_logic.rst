Temporal Logic (G, F, X) on Time-Series
========================================

This notebook demonstrates how JLNN handles Linear Temporal Logic (LTL) operators over interval-valued time-series. We focus on three fundamental operators:

* **G (Globally/Always):** The property must hold for all steps in the time window.
* **F (Eventually/Finally):** The property must hold for at least one step in the time window.
* **X (Next):** The property must hold in the very next time step.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_temporal_logic.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_temporal_logic.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.


Theoretical Background
----------------------

Traditional **Linear Temporal Logic (LTL)** evaluates formulas as strictly *True* or *False* over discrete time steps. In the **Logical Neural Network (LNN)** framework, we extend this to **Interval-Valued Fuzzy Logic**, where each temporal operator propagates uncertainty through time.

1. **The Meaning of Truth Intervals [L, U]**
   Instead of a single scalar, JLNN tracks a lower bound (*L*) and an upper bound (*U*) for every time step. The gap represents the model's uncertainty.

2. **Temporal Operators as Temporal Aggregators**
   In JLNN, temporal operators are implemented as generalized logical gates acting along the axis of time:

   * **Globally** (:math:`\mathcal{G} \phi`): Equivalent to a **generalized conjunction (AND)**. We use the Gödel t-norm (minimum). If one single moment is "False", the whole window tends toward False.
   * **Eventually** (:math:`\mathcal{F} \phi`): Equivalent to a **generalized disjunction (OR)**. We use the Gödel t-conorm (maximum). A single "True" signal is enough to satisfy the condition.
   * **Next** (:math:`\mathcal{X} \phi`): A simple temporal shift, looking at the truth value of the following step.

3. **JLNN vs. RNN/LSTM**
   Unlike "black-box" recurrent networks, JLNN temporal operators are **Explainable** (traceable logic), **Deterministic** (strict semantics), and **Uncertainty-aware**.


Content Overview
-----------------

The following snippet demonstrates how to ground raw temperature data into fuzzy predicates and evaluate temporal formulas using a sliding window approach.


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

    from jlnn.symbolic.compiler import LNNFormula
    import jax.numpy as jnp
    from flax import nnx
    import matplotlib.pyplot as plt

    time_steps = 30
    t = jnp.arange(time_steps)

    temperatures = 22 + 14 * jnp.exp(-(t - 17)**2 / 15) + jnp.sin(t*0.5) * 1.5

    def ground_high_temp(temp):
        # If temp < 25 -> False, If temp > 35 -> True
        L = jnp.clip((temp - 30) / 5, 0.0, 1.0)
        U = jnp.clip((temp - 25) / 5, 0.0, 1.0)
        return jnp.array([[L, U]])

    high_temp_inputs = jnp.stack([ground_high_temp(temp) for temp in temperatures])
    print(f"Inputs shape: {high_temp_inputs.shape}") # (30, 1, 2)

    model_G = LNNFormula("G(high_temp)", nnx.Rngs(42)) # Always
    model_F = LNNFormula("F(high_temp)", nnx.Rngs(42)) # Eventually
    model_X = LNNFormula("X(high_temp)", nnx.Rngs(42)) # Next

    window_size = 5

    def run_temporal_analysis(model, inputs, window):
        results = []
        
        # 1. OBTAINING A REAL LOGICAL NODE
        # If model.root is of type lark.Tree, the node is in children[0]
        # If it's already Node, we'll use it directly
        if hasattr(model.root, 'children'):
            root_node = model.root.children[0]
        else:
            root_node = model.root

        for i in range(len(inputs) - window + 1):
            data_window = inputs[i : i + window]
            current_inputs = {"high_temp": data_window}
            
            # 2. CALL FORWARD
            # Now we call the method on AlwaysNode / EventuallyNode
            output = root_node.forward(current_inputs)
            
            # 3. RESULT EXTRACTION
            # output is a JAX array, we take the last time step
            res = output.reshape(-1, 2)
            L, U = float(res[-1, 0]), float(res[-1, 1])
            
            results.append((L, U))
            
        return jnp.array(results)

    G_res = run_temporal_analysis(model_G, high_temp_inputs, window_size)
    F_res = run_temporal_analysis(model_F, high_temp_inputs, window_size)
    X_res = run_temporal_analysis(model_X, high_temp_inputs, window_size)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    # 1. Raw Data
    axes[0].plot(t, temperatures, 'r-o', label="Raw Temperature (°C)")
    axes[0].axhline(30, color='black', linestyle='--', alpha=0.5, label="Threshold")
    axes[0].set_title("Input: Server Room Temperature")
    axes[0].legend()

    # 2. Globally (G)
    t_res = jnp.arange(len(G_res))
    axes[1].fill_between(t_res, G_res[:,0], G_res[:,1], color='blue', alpha=0.3, label="G(high_temp)")
    axes[1].set_title("G (Always): True only if ALL steps in window are high")
    axes[1].set_ylim(-0.1, 1.1)

    # 3. Eventually (F)
    axes[2].fill_between(t_res, F_res[:,0], F_res[:,1], color='green', alpha=0.3, label="F(high_temp)")
    axes[2].set_title("F (Eventually): True if AT LEAST ONE step in window is high")
    axes[2].set_ylim(-0.1, 1.1)

    # 4. Next (X)
    axes[3].fill_between(t_res, X_res[:,0], X_res[:,1], color='purple', alpha=0.3, label="X(high_temp)")
    axes[3].set_title("X (Next): Truth value of the following step")
    axes[3].set_ylim(-0.1, 1.1)

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("Truth [L, U]")

    plt.tight_layout()
    plt.show()


Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_basic_boolean_gates.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_temporal_logic.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.


