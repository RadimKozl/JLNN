JLNN + Vision Transformer: Neuro-symbolic Image Classification
==============================================================

This tutorial demonstrates the integration of a **Vision Transformer (ViT)** backbone with a **Justifiable Logical Neural Network (JLNN)** layer. This hybrid architecture bridges the gap between high-performance visual feature extraction and interpretable logical reasoning.

.. note::
    The interactive notebook and pre-trained weights are hosted externally to ensure the best 
    viewing experience and to allow immediate execution or deployment.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
        :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_vit_training.ipynb
        :link-type: url

        Execute the from-scratch training on CIFAR-10 directly in your browser.

    .. grid-item-card::  View on GitHub
        :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_vit_training_github.ipynb
        :link-type: url

        View source code, logic monitoring graphs, and training outputs.


.. grid:: 2

    .. grid-item-card::  Weights (Kaggle)
        :link: https://www.kaggle.com/models/radimkzl/jlnn-ns-vit/
        :link-type: url
        :text-align: center
        :shadow: md

        Download the NS-ViT weights directly from the Kaggle Model Hub.

    .. grid-item-card::  Weights (Hugging Face)
        :link: https://huggingface.co/KRadim/vit-jlnn-cifar10/
        :link-type: url
        :text-align: center
        :shadow: md

        Access the model weights and configuration on the Hugging Face Hub.


The Vision: Transparent Vision Transformers
----------------------------------------------

While Vision Transformers (ViT) excel at capturing global dependencies in images, they remain "black boxes". This tutorial demonstrates a **Neuro-symbolic Vision System** where the ViT acts as a sensory organ, while the JLNN layer acts as the reasoning mind.

By mapping transformer embeddings to fuzzy predicates, we can audit the model's decision process: Is it a bird because it has "wings" and a "beak", or just because of the blue background?

The Architecture: From Pixels to Logic
-----------------------------------------

The model processes images through three distinct stages of abstraction:

1. **ViT Backbone**: A Vision Transformer (trained from scratch) extracts high-level semantic features from the CLS token.
2. **Fuzzy Grounding**: A specialized layer with temperature scaling (:math:`\tau=1.4`) and centered bias (:math:`b=-1.2`) that maps continuous features into logical predicates.
3. **JLNN Layer**: Implements Łukasiewicz t-norm logic to evaluate human-defined rules, providing a classification along with a logical audit trail.

Core Symbolic Rules (JLNN Syntax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model doesn't just predict a class ID; it evaluates structured hypotheses. For example, the definition of an animal in our logical space:

.. code-block:: python

   # Rule 0: The Animal Hypothesis
   "0.75 :: (body & head & eyes & mouth) -> is_animal"

Key Features
---------------

* **From-Scratch Training**: Demonstrates that a Transformer-Logic hybrid can converge stably without pre-trained ImageNet weights.
* **Explainable AI (XAI)**: Every prediction produces an audit trail of which visual parts triggered which logical rule.
* **Uncertainty Quantification**: The JLNN layer naturally handles and propagates uncertainty using :math:`[L, U]` truth intervals.

Implementation Details
-------------------------

The pipeline is optimized for the **JAX/Flax NNX** ecosystem. Key components include:

Fuzzy Grounding with Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To prevent "binary collapse" (where predicates become stuck at 0 or 1), we utilize a calibrated grounding layer:

.. code-block:: python

    # Grounding with temperature scaling and bias for stable convergence
    grounding = FuzzyGrounding(
        n_features=192, 
        n_predicates=len(predicates),
        tau=1.4, 
        bias_init=-1.2
    )

Output Structure
~~~~~~~~~~~~~~~~~~~

The model's ``__call__`` method is designed for auditing, returning a nested structure:

.. code-block:: python

    # output[0] -> Logical Audit ([L, U] intervals for rules)
    # output[1] -> Grounded Predicates (fuzzy truth of visual parts)
    # output[2] -> Classification Logits (raw scores for classes)
    audit, predicates, logits = model(image_batch)

All example code
------------------

.. code-block:: python

    # Note: Full implementation is available in the linked notebook.
    # Below is a conceptual snippet of the model definition.

    import jax
    from flax import nnx
    from jlnn.nn.layers import JLNNLayer, FuzzyGrounding

    class ViT_JLNN(nnx.Module):
        def __init__(self, vit_backbone, rules, rngs):
            self.backbone = vit_backbone
            self.grounding = FuzzyGrounding(192, n_predicates, tau=1.4, bias_init=-1.2)
            self.logic = JLNNLayer(rules, rngs)

        def __call__(self, x):
            # 1. Feature Extraction (ViT)
            features = self.backbone(x) # CLS token
            
            # 2. Symbol Grounding
            z = self.grounding(features)
            
            # 3. Logical Inference
            audit = self.logic(z)
            
            # 4. Final Classification Head
            logits = self.create_logits(audit)
            
            return audit, z, logits

Interpreting the Audit
-------------------------

By evaluating these rules, the model provides **Justifiable Predictions**. If the truth values (:math:`[L, U]` intervals) for a rule are narrow (e.g., `[0.85, 0.90]`), the model is confident in its logical reasoning. A wide interval (e.g., `[0.10, 0.90]`) indicates that the visual evidence is insufficient to satisfy the symbolic constraints.

Download
-----------

You can download the raw notebook file or the pre-trained weights:

* :download:`JLNN_vit_training.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_vit_training.ipynb>`
* **Weights (Kaggle)**: `JLNN_NS_ViT <https://www.kaggle.com/models/radimkzl/jlnn-ns-vit/>`_
* **Weights (Hugging Face)**: `KRadim/vit-jlnn-cifar10 <https://huggingface.co/KRadim/vit-jlnn-cifar10/>`_