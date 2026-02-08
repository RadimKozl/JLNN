Base Example: Basic inference and manual grounding
======================================================

This tutorial presents the simplest way to use the **JLNN** framework for logical reasoning with uncertainty. 
It focuses on defining rules, manually setting truth intervals, 
and calculating the resulting inference without the need to train a model.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_basic_inference.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_basic_inference.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

Key concepts
---------------

In JLNN, we work with logical rules that are represented as 
a differentiable graph (NNX). Instead of a single truth value, we work with an **interval [L, U]**:

* **L (Lower bound):** Minimum confirmed truth.
* **U (Upper bound):** Maximum possible truth.
* **Interval width (U - L):** Expresses the degree of uncertainty or ignorance about the given predicate.

Rule definition
------------------

We create the model by compiling a symbolic rule using the ``LNNFormula`` class. 
In this example, we consider an implication with a weight of 0.8:

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
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''
    
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"

    import jax.numpy as jnp
    from flax import nnx
    from jlnn.symbolic.compiler import LNNFormula

    print("JLNN loaded.")

    # Model creation – compiling rules into an NNX graph
    model = LNNFormula("0.8::A & B -> C", nnx.Rngs(42))

    print("Model created. Predicates:", list(model.predicates.keys()))

    # Manual grounding – setting intervals

    inputs = {
        "A": jnp.array([[0.7, 0.9]]),   # And it is quite likely
        "B": jnp.array([[0.4, 0.8]]),   # B has more uncertainty
        "C": jnp.array([[0.0, 1.0]])    # C is completely unknown (ignorance)
    }

    output = model(inputs)

    print("Output shape:", output.shape)

    if len(output.shape) == 3:          # (batch, 1, 2) or similar
        L = output[0, 0, 0].item()
        U = output[0, 0, 1].item()
    elif len(output.shape) == 2:        # (batch, 2)
        L = output[0, 0].item()
        U = output[0, 1].item()
    elif len(output.shape) == 1:        # only (2,)
        L = output[0].item()
        U = output[1].item()
    else:
        raise ValueError(f"Unknown output shape: {output.shape}")

    print("Output interval for C:")
    print(f"  L = {L:.4f}")
    print(f"  U = {U:.4f}")
    print(f"  Uncertainty (width): {U - L:.4f}")

    # Experiments – different input intervals

    inputs_exp1 = {
        "A": jnp.array([[0.95, 1.0]]),
        "B": jnp.array([[0.90, 0.98]]),
        "C": jnp.array([[0.0, 1.0]])
    }
    print("Exp 1 – strong A and B:")
    print(model(inputs_exp1))

    inputs_exp2 = {
        "A": jnp.array([[0.95, 1.0]]),
        "B": jnp.array([[0.1, 0.3]]),
        "C": jnp.array([[0.0, 1.0]])
    }
    print("\nExp 2 – weak B:")
    print(model(inputs_exp2))

    inputs_exp3 = {
        "A": jnp.array([[0.4, 0.9]]),
        "B": jnp.array([[0.8, 0.95]]),
        "C": jnp.array([[0.0, 1.0]])
    }
    print("\nExp 3 – high uncertainty in A:")
    print(model(inputs_exp3))



Tutorial summary
------------------

* **Symbols to Graph:** JLNN converts logic rules to NNX graphs.
* **Interval Logic:** We work with intervals [L, U], not fixed points.
* **No Training:** In this mode, all parameters are set manually and the model is used for direct logical inference.

Download
--------

You can also download the raw notebook file for local use:
:download:`JLNN_basic_inference.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_basic_inference.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.