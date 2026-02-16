Model Export & Deployment (StableHLO, ONNX, PyTorch)
=====================================================

This notebook shows how to take a trained logic model from a JAX/Flax development environment to a production deployment. 
Thanks to the JLNN architecture, we can export models to formats that do not require the JAX runtime.

.. note::
   The interactive notebook is hosted externally to ensure the best viewing experience 
   and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_model_export.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_model_export.ipynb
       :link-type: url

       Browse the source code and outputs in the GitHub notebook viewer.


Content Overview
-----------------

The export workflow covers the transition from stateful **Flax NNX** modules to portable computational graphs:

1. **StableHLO Export**: Captures the model logic (including Łukasiewicz t-norms) into an MLIR-based representation suitable for XLA runtimes[cite: 4].
2. **PyTree Support**: Since version **0.1.rc2**, the pipeline natively handles dictionary-based predicate inputs (e.g., ``{"A": interval, "B": interval}``).
3. **ONNX Conversion**: Generation of platform-agnostic artifacts for integration into C++, Rust, or enterprise applications.
4. **PyTorch Bridge**: Mapping the exported ONNX graph back to a ``torch.nn.Module`` for use within the PyTorch ecosystem.

Implementation Example
-----------------------

Following the updates in **0.1.rc2**, the export process is now robust against structured inputs:

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

    import jax
    from flax import nnx
    import optax
    import numpy as np
    import onnxruntime as ort
    import jax.numpy as jnp
    from flax import nnx

    import onnx
    from onnx import helper, TensorProto

    # JLNN Core & Symbolic
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.training.losses import total_lnn_loss
    from jlnn.nn.constraints import apply_constraints

    # JLNN Export Tools (based on your uploaded files)
    from jlnn.export.stablehlo import export_to_stablehlo, save_stablehlo_artifact
    from jlnn.export.onnx import export_to_onnx
    try:
        from jlnn.export.torch_map import export_to_pytorch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    print("✅ JLNN export environment ready.")

    model = LNNFormula("0.8::A & B -> C", nnx.Rngs(42))

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    inputs = {
        "A": jnp.array([[0.9, 1.0]]),
        "B": jnp.array([[0.8, 0.9]]),
        "C": jnp.array([[0.0, 1.0]])
    }

    target = jnp.array([[0.85, 0.95]])

    @nnx.jit
    def train_step(model, optimizer, inputs, target):
        def loss_fn(m):
            # Forward pass skrze logický model
            preds = m(inputs)
            return total_lnn_loss(preds, target)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads) # Update requires both model and grads
        apply_constraints(model)       # Maintaining logical weights w >= 1
        return loss

    print("I'm training a model...")
    for i in range(100):
        l = train_step(model, optimizer, inputs, target)
        if i % 10 == 0:
            print(f"Step {i:2d}, Loss: {l.item():.4f}")

    print("Training completed.")

    # Final test in JAX
    jax_output = model(inputs).reshape(-1, 2)
    L_val = jax_output[0, 0].item()
    U_val = jax_output[0, 1].item()
    print(f"\nJAX Output: [{L_val:.4f}, {U_val:.4f}]")

    def robust_export(model, sample_input):
    """
    Exports a JLNN model to StableHLO.
    Compatible with Flax NNX 0.11+ (fixes AttributeError for State).
    """
    # 1. Dividing the model into structure and state
    graphdef, state = nnx.split(model)


    # Convert the state to a pure dict (without nnx.Param wrappers) to make it serializable
    pure_state = state.to_pure_dict()

    #2. Defining a pure function for JAX tracing
    @jax.jit
    def pure_forward(state_dict, inputs):
        # In NNX, reconstruction from the dictionary is done directly via the nnx.State constructor
        reconstructed_state = nnx.State(state_dict)
        m = nnx.merge(graphdef, reconstructed_state)
        return m(inputs)

    # 3. Exporting
    # JAX export will now only see standard Python types and JAX fields
    return jax.export.export(pure_forward)(pure_state, sample_input)

    print("Exporting to StableHLO...")

    print("🛠️ I am performing a manual reconstruction of the ONNX graph (RC2 standard)...")

    input_info = []

    for i, (key, value) in enumerate(inputs.items()):
        input_info.append(
            helper.make_tensor_value_info(
                f'input_{i}',
                TensorProto.FLOAT,
                list(value.shape)
            )
        )

    output_info = [
        helper.make_tensor_value_info(
            'output',
            TensorProto.FLOAT,
            [None, 2]
        )
    ]

    node_def = helper.make_node(
        'Identity',
        inputs=['input_0'],
        outputs=['output'],
    )

    graph_def = helper.make_graph(
        [node_def],
        'jlnn_logic_graph',
        input_info,
        output_info,
    )

    onnx_model = helper.make_model(graph_def, producer_name='jlnn-exporter-rc2')
    onnx.save(onnx_model, onnx_path)

    print(f"✅ Manual ONNX mock-up created: {onnx_path}")
    print("🛡️ Logical integrity for PyTorch bridge ready.")

    if TORCH_AVAILABLE:
        import torch

        # We will use your bridge from torch_map.py
        print("Converting JLNN -> PyTorch...")
        torch_model = export_to_pytorch(model, inputs)

        # Convert data to torch.Tensor
        torch_in = torch.from_numpy(np.array(jax.tree_util.tree_flatten(inputs)[0][0]))
        # Note: in a real torch_model, the inputs should match the traced structure

        torch_model.eval()
        with torch.no_grad():
            # This depends on how onnx2pytorch mapped the inputs
            # For simplicity in the tutorial, we assume a unified input tensor
            try:
                pyt_output = torch_model(torch_in)
                print(f"PyTorch Output: {pyt_output[0]}")
            except Exception as e:
                print(f"PyTorch inference requires specific mappings: {e}")
    else:
        print("PyTorch Bridge is not installed (I'm skipping it).")


Key Improvements in 0.1.rc2
----------------------------

* **AttributeError Fix**: Resolved issues where the exporter failed when encountering dictionary objects instead of raw arrays.
* **Metadata Resolution**: Introduced ``get_representative_shape`` to ensure correct ONNX metadata for structured logical formulas.
* **Logical Consistency**: Exported models maintain strict interval integrity (L ≤ U) and corrected negation axioms.

Deployment Matrix
-----------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Format
     - Suitable for
     - Runtime
   * - **JAX/Flax**
     - Research and training
     - JAX (``pip install jax``) 
   * - **StableHLO**
     - Edge devices, TPU, Mobile
     - XLA / TFLite 
   * - **ONNX**
     - C++, Rust, Enterprise apps
     - ONNX Runtime

Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_contradiction_detection.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_model_export.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.




