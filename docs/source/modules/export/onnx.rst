ONNX Export
===========

.. automodule:: jlnn.export.onnx
   :members:
   :undoc-members:
   :show-inheritance:

This module ensures the transformation of JLNN models into the **Open Neural Network Exchange (ONNX)** format. 

Implementation
--------------

Unlike standard wrappers, JLNN uses the native ``jax.export`` pipeline. The export process occurs in two phases:

1. **Model Tracing**: Using ``export_to_stablehlo``, the stateful NNX model is converted into a static computational graph.
2. **Metadata Serialization**: The function ``export_to_onnx`` prepares the model for external runtime environments.

Advantages:
-----------
* **Native JAX Support**: No external dependencies like ``jax2onnx``. We use the standardized JAX interface.
* **XLA Optimizations**: The exported model contains optimizations performed by the XLA compiler.
* **Interoperability**: Ability to run JLNN logic reasoning in environments such as C++, C#, or directly in a browser.

Main Functions
--------------

.. autofunction:: export_to_onnx

   Converts a trained model into an ONNX artifact.
   
   .. code-block:: python

      # Example export
      sample_input = jnp.zeros((1, n_inputs, 2))
      export_to_onnx(model, sample_input, "logic_model.onnx")

.. autofunction:: save_for_xla_runtime