ONNX Export
===========

.. automodule:: jlnn.export.onnx
   :members:
   :undoc-members:
   :show-inheritance:

This module ensures the transformation of JLNN models into the **Open Neural Network Exchange (ONNX)** format.

Implementation
--------------
Unlike standard wrappers, JLNN uses the native ``jax.export`` pipeline.

1. **Model Tracing**: Using ``export_to_stablehlo``, the stateful NNX model is converted into a static computational graph.
2. **Metadata Serialization**: The function ``export_to_onnx`` prepares the model for external runtime environments.

.. code-block:: python

   # Example export
   sample_input = jnp.zeros((1, n_inputs, 2))
   export_to_onnx(model, sample_input, "logic_model.onnx")