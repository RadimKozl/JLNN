Contradiction Detection & Model Repair
=======================================

This notebook demonstrates a unique feature of JLNN: the ability to detect and correct logical conflicts. In classical neural networks, 
conflicting data leads to a "blurred" average, in JLNN it leads to an L > U state, which can be explicitly identified and resolved.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_contradiction_detection.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_contradiction_detection.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.


Content Overview
-----------------

This tutorial demonstrates the "Self-Healing" capability of JLNN. When logical rules conflict with observed data, the network enters a state of contradiction (L > U). 

The following example shows how to:
1. Initialize a model with a forced contradiction.
2. Use ``total_lnn_loss`` to penalize logically invalid states.
3. Apply ``apply_constraints`` to keep the model within the bounds of Łukasiewicz semantics.

Key Takeaways
--------------
* **Contradiction Detection:** Unlike black-box models, JLNN explicitly signals when it is confused by conflicting information.
* **Differentiable Logic:** The consistency constraint L ≤ U is part of the loss function, allowing the model to "unlearn" or weaken conflicting rules during training.

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

    import jax.numpy as jnp
    from flax import nnx
    import optax
    import matplotlib.pyplot as plt

    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.nn.constraints import apply_constraints
    from jlnn.training.losses import total_lnn_loss

    model = LNNFormula("0.95::A & B -> C", nnx.Rngs(42))

    inputs = {
        "A": jnp.array([[0.8, 0.9]]),
        "B": jnp.array([[0.7, 0.8]]),
        "C": jnp.array([[0.0, 0.2]]) # Target will conflict with the rule
    }

    root_node = model.root.children[0] if hasattr(model.root, 'children') else model.root
    initial_output = root_node.forward(inputs)

    flat_out = initial_output.reshape(-1, 2)

    L_init = flat_out[0, 0].item()
    U_init = flat_out[0, 1].item()

    print(f"Initial C: [{L_init:.4f}, {U_init:.4f}]")
    print(f"Contradiction detected: {L_init > U_init}")

    def plot_contradiction(L, U, title):
        fig, ax = plt.subplots(figsize=(8, 2))
        is_conflict = L > U
        color = 'salmon' if is_conflict else 'skyblue'

        # Plot the interval
        start = min(L, U)
        width = abs(U - L) if is_conflict else (U - L)

        ax.barh(['Truth Value'], [width], left=[start], color=color, height=0.5)
        ax.axvline(L, color='blue', linestyle='--', label=f'L={L:.2f}')
        ax.axvline(U, color='red', linestyle='--', label=f'U={U:.2f}')

        ax.set_xlim(-0.1, 1.1)
        ax.set_title(title)
        ax.legend(loc='lower right')
        if is_conflict:
            ax.text(0.5, 0.2, "CONTRADICTION (L > U)", color='red', fontweight='bold', ha='center')
        plt.show()

    
    optimizer = nnx.Optimizer(
        model, 
        optax.adam(0.02),
        wrt=nnx.Param  # <--- This is a key parameter
    )

    target = jnp.array([[0.0, 0.2]]) # Target interval for C (low true)

    @nnx.jit
    def train_step(model, optimizer, inputs, target):
        def loss_fn(m):
            # Forward pass skrze logický uzel
            node = m.root.children[0] if hasattr(m.root, 'children') else m.root
            pred = node.forward(inputs)
            # Using the total_lnn_loss function
            return total_lnn_loss(pred, target)

        # Calculate loss and gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        
        # FLAX 0.11+: update now requires both model and grads
        optimizer.update(model, grads)
        
        # Applying logical constraints (weights >= 1, L <= U)
        apply_constraints(model)
        
        return loss

    print("=== Repairing model consistency ===")
    for step in range(101):
        current_loss = train_step(model, optimizer, inputs, target)
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {current_loss:.6f}")

    final_output = root_node.forward(inputs)

    flat_final = final_output.reshape(-1, 2)

    L_final = flat_final[0, 0].item()
    U_final = flat_final[0, 1].item()

    print(f"Fixed C: [{L_final:.4f}, {U_final:.4f}]")
    print(f"Logical conflict removed? {L_final <= U_final}")

    plot_contradiction(L_final, U_final, "State After Repair (Consistency Restored)")


Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_contradiction_detection.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_contradiction_detection.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.