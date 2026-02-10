Basic Boolean Gates
====================

This tutorial demonstrates how JLNN handles classical Boolean operations 
(AND, OR, NOT, NAND, NOR, XOR) using interval logic. You will see how 
crisp inputs (0/1) translate to certain intervals and how the system 
manages uncertainty.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_basic_boolean_gates.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_basic_boolean_gates.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.

Content Overview
-----------------

This tutorial explores the intersection of classical logic and neuro-symbolic reasoning. You will learn how to:

* **Define Boolean Gates as Logical Formulas**: Using JLNN's symbolic syntax to represent AND, OR, NOT, NAND, NOR, and XOR operations.
* **Bridge Crisp and Fuzzy Inputs**: Observe how the system handles perfect certainty (0/1) versus real-world uncertainty (e.g., "almost true" intervals like [0.95, 1.0]).
* **Monitor Interval Integrity**: Verify that the system maintains consistent boundaries ($L \le U$) even through complex nested operations like XOR.
* **Analyze Uncertainty Widths**: Visualize how "ignorance" propagates through different logical gates using JAX-backed computations.


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
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"

    import jlnn
    import jax.numpy as jnp
    from flax import nnx
    import jax
    import matplotlib.pyplot as plt

    # JLNN core imports
    from jlnn.nn.gates import WeightedAnd, WeightedOr, WeightedNot, WeightedNand, WeightedNor, WeightedXor
    from jlnn.nn.predicates import FixedPredicate
    from jlnn.symbolic.compiler import LNNFormula, PredicateNode, UnaryGateNode, BinaryGateNode

    print("JLNN loaded.")

    crisp_inputs = {
        "A": jnp.array([[1.0, 1.0]]),   # A = True
        "B": jnp.array([[0.0, 0.0]])    # B = False
    }

    fuzzy_inputs = {
        "A": jnp.array([[0.95, 1.0]]),  # A almost True
        "B": jnp.array([[0.05, 0.1]])   # B almost False
    }

    def run_gate(rule, inputs):
        model = LNNFormula(rule, nnx.Rngs(42))
        
        # Key override: replace LearnedPredicate with Fixed (identity)
        for name in inputs:
            if name in model.predicates:
                model.predicates[name].predicate = FixedPredicate()
        
        output = model(inputs)
        
        # Fuse L <= U (although it should already be OK)
        output = jnp.sort(output, axis=-1)
        
        L = output[0, 0].item()
        U = output[0, 1].item()
        width = U - L
        
        print(f"Rule: {rule}")
        print(f"  Output: [{L:.4f}, {U:.4f}] (width {width:.4f})")
        
        return L, U, width

    
    print("=== AND ===")
    run_gate("A & B", crisp_inputs)   # we expect [0,0]
    run_gate("A & B", fuzzy_inputs)   # we expect low values ​​+ small width

    print("=== OR ===")
    run_gate("A | B", crisp_inputs)   # we expect [1,1]
    run_gate("A | B", fuzzy_inputs)   # we expect high values

    print("=== NOT ===")
    run_gate("~A", {"A": crisp_inputs["A"]})     # we expect [0,0]
    run_gate("~A", {"A": fuzzy_inputs["A"]})     # we expect low values

    print("=== NOT ===")
    run_gate("~B", {"B": crisp_inputs["B"]})
    run_gate("~B", {"B": fuzzy_inputs["B"]})

    print("=== NAND ===")
    run_gate("~(A & B)", crisp_inputs)

    print("=== NOR ===")
    run_gate("~(A | B)", crisp_inputs)

    print("=== XOR ===")
    run_gate("(A & ~B) | (~A & B)", crisp_inputs)
    run_gate("(A & ~B) | (~A & B)", fuzzy_inputs)   # interesting – XOR is sensitive to uncertainty

    gates = ["AND", "OR", "NOT A", "NAND", "NOR", "XOR"]
    crisp_widths = []
    fuzzy_widths = []

    for gate in gates:
        # Crisp
        if gate == "NOT A":
            rule = "~A"
            inp = {"A": crisp_inputs["A"]}
        elif gate == "XOR":
            rule = "(A & ~B) | (~A & B)"
            inp = crisp_inputs
        else:
            rule = f"A {'&' if 'AND' in gate or 'NAND' in gate else '|'} B"
            if 'NAND' in gate or 'NOR' in gate:
                rule = f"~({rule})"
            inp = crisp_inputs
        
        _, _, w_crisp = run_gate(rule, inp)
        crisp_widths.append(w_crisp)
        
        # Fuzzy – we correctly pass the keys from inp
        inp_f = {key: fuzzy_inputs[key] for key in inp}
        _, _, w_fuzzy = run_gate(rule, inp_f)
        fuzzy_widths.append(w_fuzzy)

    # Check lengths (debug)
    print(f"Gates: {len(gates)}")
    print(f"Crisp widths: {len(crisp_widths)}")
    print(f"Fuzzy widths: {len(fuzzy_widths)}")

    # If duplicates – take only the last unique values ​​(if the loop ran twice)
    if len(crisp_widths) > len(gates):
        crisp_widths = crisp_widths[-len(gates):]
        fuzzy_widths = fuzzy_widths[-len(gates):]

    x = range(len(gates))
    bar_width = 0.35

    # Fuzzy – full columns
    plt.bar([i + bar_width/2 for i in x], fuzzy_widths, width=bar_width, label='Fuzzy inputs', color='orange')

    # Crisp – thin columns or just dots at zero
    for i, w in enumerate(crisp_widths):
        if w > 0:
            plt.bar(i - bar_width/2, w, width=bar_width, color='skyblue', label='Crisp inputs' if i == 0 else "")
        else:
            plt.plot(i - bar_width/2, 0, 'o', color='skyblue', markersize=10, label='Crisp inputs (width=0)' if i == 0 else "")

    plt.xticks(x, gates, rotation=45)
    plt.ylabel('Uncertainty width (U - L)')
    plt.title('Output uncertainty for various Boolean operations')
    plt.legend()
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.figtext(0.5, -0.15, 
                "Note: Crisp inputs (0/1) always have width 0 → blue columns are invisible (exact logic).",
                ha='center', fontsize=10, color='gray')
    plt.show()

Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_basic_boolean_gates.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_basic_boolean_gates.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.