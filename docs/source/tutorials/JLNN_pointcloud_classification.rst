3D Point Cloud Classification with JLNN
==========================================

This tutorial demonstrates how to integrate raw 3D spatial data with symbolic reasoning using the **JLNN (JAX Logical Neural Networks)** framework. We use a PointNet-style encoder to extract geometric features and a logical layer to classify objects based on human-interpretable rules.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_pointcloud_classification.ipynb
       :link-type: url

       Run the autonomous optimization cycle in your browser.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_pointcloud_classification.ipynb
       :link-type: url

       Browse the full source code and results.


Introduction
---------------

Traditional 3D deep learning models act as "black boxes". By using JLNN, we can define objects using logical constraints:

* A **Chair** must have a backrest and legs.
* A **Table** has a large horizontal plane and legs, but *no* backrest.
* A **Sofa** shares features with a chair but is typically larger in scale.

Model Architecture
---------------------

The architecture consists of two main components:

1. **Neural Feature Extractor (PointNet):** Consists of shared multi-layer perceptrons (MLPs) and a global max-pooling layer to extract a global feature vector from raw *(N, 3)* point coordinates.

2. **Symbolic Logic Layer (JLNN):** A differentiable logic circuit that maps neural activations to truth intervals *[L, U]*.

.. code-block:: python

    class NSModelNet(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.encoder = PointNetEncoder(rngs)
            self.to_predicates = nnx.Linear(128, 6, rngs=rngs)
            
            # Logical definitions of ModelNet classes
            self.rules = nnx.Dict({
                'Chair': LNNFormula("HasHorizontalPlane & HasLegs & HasBackrest", rngs),
                'Table': LNNFormula("LargeHorizontalPlane & HasLegs & ~HasBackrest", rngs),
                'Sofa':  LNNFormula("LargeHorizontalPlane & HasBackrest", rngs)
            })

Training Strategy
-------------------

To ensure the neuro-symbolic bridge learns effectively, we use a hybrid loss function and gradient clipping.

**Loss Function:**
We combine Binary Cross Entropy (BCE) on the lower bounds *(L)* with an uncertainty penalty to push the truth intervals towards crisp values.

.. math::

    \mathcal{L} = \text{BCE}(L, y) + \lambda \sum_{i} (U_i - L_i)

**Gradient Clipping:**
Since logical operators (AND/OR) can introduce sharp non-linearities, global norm clipping is essential for stable training.

.. code-block:: python

    # Update step with Flax NNX 0.11+ API
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    optimizer.update(model, grads)

Interpreting Results
----------------------

One of the key advantages of JLNN is transparency. Instead of a simple probability score, we get a truth interval.

**Example Analysis:**

.. list-table:: Object Analysis (Label: Chair)
   :widths: 25 25 25
   :header-rows: 1

   * - Class
     - Lower Bound (L)
     - Upper Bound (U)
   * - **Chair**
     - **0.2242**
     - **0.3044**
   * - Sofa
     - 0.1550
     - 0.2545
   * - Table
     - 0.0751
     - 0.0960

In this example, the model correctly identifies the **Chair**. The non-zero score for **Sofa** indicates that the model correctly identified shared features (like the backrest) but ultimately favored the chair definition.

Tutorial code
---------------

.. code-block:: python

    '''
    try:
      import jlnn
      import jraph
      import numpyro
      import optax
      import trimesh
      from flax import nnx
      import jax.numpy as jnp
      import networkx as nx
      import numpy as np
      import xarray as xr
      import pandas as pd
      import open3d as o3d
      import grain.python as grain
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
      !pip install grain --quiet
      !pip install open3d --quiet
      !pip install networkx --quiet
      !pip install trimesh --quiet
      !pip install xarray --quiet
      !pip install kagglehub --quiet
      !pip install optax --quiet
      # Fix JAX/CUDA compatibility for 2026 in Colab
      !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
      !pip install  scikit-learn pandas --quiet

      import os
      print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
      os.kill(os.getpid(), 9)
      os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    # Imports

    import os
    import optax
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import trimesh
    import kagglehub
    import open3d as o3d
    import numpy as np
    import matplotlib.pyplot as plt
    import grain.python as grain
    from sklearn.cluster import DBSCAN
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.nn.functional import weighted_and

    # Downloading the Modelnet10 data file.

    # Download latest version
    path = kagglehub.dataset_download("balraj98/modelnet10-princeton-3d-object-dataset")

    print("Path to dataset files:", path)

    # ModelNet10 is in the 'ModelNet10' subdirectory
    data_root = os.path.join(path, 'ModelNet10')

    # Grain Data Pipeline
    class ModelNet10Source(grain.RandomAccessDataSource):
      def __init__(self, root_dir, split='train'):
          self.files = []
          self.labels = []
          self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
          self.class_map = {name: i for i, name in enumerate(self.class_names)}

          for cls in self.class_names:
              cls_dir = os.path.join(root_dir, cls, split)
              if os.path.exists(cls_dir):
                  for f in os.listdir(cls_dir):
                      if f.endswith('.off'):
                          self.files.append(os.path.join(cls_dir, f))
                          self.labels.append(self.class_map[cls])

      def __getitem__(self, idx):
          mesh = trimesh.load(self.files[idx])
          if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)
          points = mesh.sample(1024).astype(np.float32)
          # Normalization
          points -= np.mean(points, axis=0)
          points /= np.max(np.linalg.norm(points, axis=1))
          return {"points": points, "label": self.labels[idx]}

      def __len__(self): return len(self.files)

    # Loader initialization

    source = ModelNet10Source(data_root)
    dataset = grain.MapDataset.source(source).shuffle(seed=42).batch(16)
    data_iter = dataset.to_iter_dataset()

    # Neuro-Symbolic Model (PointNet + JLNN)

    class NSModelNet(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
          self.conv1 = nnx.Linear(3, 64, rngs=rngs)
          self.conv2 = nnx.Linear(64, 128, rngs=rngs)
          # 8 predicates (more than before – covers all classes)
          self.to_predicates = nnx.Linear(128, 8, rngs=rngs)

          # Rules for ALL 10 ModelNet10 classes
          # (bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet)
          self.rules = nnx.Dict({
              'bathtub':     LNNFormula("HasCurvedSurface & HasConcavity & ~HasLegs", rngs),
              'bed':         LNNFormula("LargeHorizontalPlane & HasVerticalSupport & ~HasLegs", rngs),
              'chair':       LNNFormula("HasHorizontalPlane & HasVerticalSupport & HasLegs & HasBackrest", rngs),
              'desk':        LNNFormula("LargeHorizontalPlane & HasLegs & ~HasBackrest & ~HasConcavity", rngs),
              'dresser':     LNNFormula("HasVerticalSupport & HasMultipleLayers & ~HasLegs", rngs),
              'monitor':     LNNFormula("HasVerticalSupport & HasThinProfile & ~HasHorizontalPlane", rngs),
              'night_stand': LNNFormula("HasHorizontalPlane & HasLegs & ~LargeHorizontalPlane", rngs),
              'sofa':        LNNFormula("LargeHorizontalPlane & HasBackrest & HasVerticalSupport", rngs),
              'table':       LNNFormula("LargeHorizontalPlane & HasLegs & ~HasBackrest", rngs),
              'toilet':      LNNFormula("HasCurvedSurface & HasVerticalSupport & HasConcavity", rngs),
          })

      def __call__(self, x):
          x = jax.nn.relu(self.conv1(x))
          x = jax.nn.relu(self.conv2(x))
          x = jnp.max(x, axis=1)  # Global Max Pooling

          # 8 predicates → sigmoid to [0, 1]
          raw_p = jax.nn.sigmoid(self.to_predicates(x))

          input_dict = {
              "HasHorizontalPlane":   raw_p[:, 0:1],
              "HasVerticalSupport":   raw_p[:, 1:2],
              "LargeHorizontalPlane": raw_p[:, 2:3],
              "HasLegs":              raw_p[:, 3:4],
              "HasBackrest":          raw_p[:, 4:5],
              "HasCurvedSurface":     raw_p[:, 5:6],
              "HasConcavity":         raw_p[:, 6:7],
              # shared predicates for dresser/monitor/night_stand
              "HasMultipleLayers":    raw_p[:, 7:8],
              "HasThinProfile":       raw_p[:, 7:8],  # same neuron, different semantics
          }

          return {name: rule(input_dict) for name, rule in self.rules.items()}
    
    # Learning Loop

    # Auxiliary loss_fn (reference, not called directly – the version inside train_step is used)
    def loss_fn_ref(model, batch, class_names):
        """
        We map targets via actual indexes from the dataset, not via the order in sorted(rules).
        class_names = source.class_names (alphabetical sorted ModelNet10 classes)
        """
        preds = model(batch['points'])
        loss = 0.0
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        for cls_name, res in preds.items():
            true_idx = class_to_idx.get(cls_name, -1)
            if true_idx == -1:
                continue  # class not in dataset → skip
            target = (batch['label'] == true_idx).astype(jnp.float32)
            loss += jnp.mean((res[:, 0] - target) ** 2)
        return loss
    
    # Initialization

    rngs = nnx.Rngs(42)
    model = NSModelNet(rngs)
    # More stable optimizer
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # train_step with correct mapping of classes to labels
    # 'class_names' is a closure over the variable source.class_names defined in Cell 11

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            preds_dict = model(batch['points'])
            current_batch_size = batch['label'].shape[0]
            epsilon = 1e-7

            # --- (inside jit): static mapping class→label index ---
            # The order must match ModelNet10 alphabetical sorting:
            # bathtub=0, bed=1, chair=2, desk=3, dresser=4,
            # monitor=5, night_stand=6, sofa=7, table=8, toilet=9
            class_label_map = {
                'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4,
                'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9,
            }

            bce = 0.0
            uncertainty = 0.0
            n_classes = len(preds_dict)

            for cls_name, res in preds_dict.items():
                true_idx = class_label_map[cls_name]
                # Target shape: (current_batch_size,)
                target = (batch['label'][:current_batch_size] == true_idx).astype(jnp.float32)

                # Lower bound (truthfulness)
                lower = jnp.ravel(res[:, 0])[:current_batch_size]

                # BCE loss
                bce += -jnp.mean(
                    target * jnp.log(lower + epsilon) +
                    (1.0 - target) * jnp.log(1.0 - lower + epsilon)
                )

                # Uncertainty penalty: penalizes large interval [L, U]
                upper = jnp.ravel(res[:, 1])[:current_batch_size]
                uncertainty += jnp.mean(upper - lower)

            bce = bce / n_classes
            uncertainty = (uncertainty / n_classes) * 0.05  # regularization weight

            total_loss = bce + uncertainty
            return total_loss, (bce, uncertainty)

        (loss, (bce, unc)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Gradient clipping – stabilizes training
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        optimizer.update(model, grads)
        return loss, bce, unc

    # Demo training

    history_loss = []
    history_bce = []
    history_unc = []

    print("🚀 Starting training...")

    for i, batch in enumerate(data_iter):
        # Calling your fixed train_step(model, optimizer, batch)
        loss, bce, unc = train_step(model, optimizer, batch)

        history_loss.append(float(loss))
        history_bce.append(float(bce))
        history_unc.append(float(unc))

        if i % 100 == 0:
            print(f"Batch {i:4d} | Loss: {loss:.4f} | BCE: {bce:.4f} | Unc: {unc:.4f}")

        if i >= 2500: # Now give it time, the model has finally started moving
            break

    print("✅ Training finished.")

    # Result: Interpretable Decision

    # 1. We create a real iterator from the Grain dataset
    it = iter(data_iter)

    # 2. Now we can use next()
    test_batch = next(it)

    sample_idx = 0
    predictions = model(test_batch['points'])

    class_label_map = {
        'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4,
        'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9,
    }

    print(f"\n--- Object Analysis ---")
    real_label = int(test_batch['label'][0])
    real_name = source.class_names[real_label]
    print(f"Actual label: {real_label} ({real_name})")
    print("-" * 60)

    best_cls = max(predictions.items(), key=lambda kv: float(jnp.ravel(kv[1][0])[0]))

    for cls, res in sorted(predictions.items()):
        data = jnp.ravel(res[0])
        l, u = float(data[0]), float(data[1])
        true_idx = class_label_map[cls]
        is_correct = (true_idx == real_label)
        is_best    = (cls == best_cls[0])
        flag = "✅" if (is_best and is_correct) else ("❌" if is_best else "  ")
        print(f"{flag} {cls:<12} | L: {l:.4f}  U: {u:.4f}")

    predicted_name = best_cls[0]
    print(f"\n→ Model predicts: {predicted_name} | Correct: {predicted_name == real_name}")

    # Visualization of Loss Curve

    def plot_loss(losses):
      plt.figure(figsize=(10, 5))

      # Plotting raw data
      plt.plot(losses, alpha=0.3, color='royalblue', label='Raw Loss')

      # Calculation and plotting of the smoothed trend (moving average over 5 batches)
      if len(losses) > 5:
          smooth_loss = [sum(losses[max(0, i-5):i+1]) / len(losses[max(0, i-5):i+1]) for i in range(len(losses))]
          plt.plot(smooth_loss, color='red', linewidth=2, label='Smoothed Trend')

      plt.title('JLNN Training Progress: ModelNet10 Loss')
      plt.xlabel('Batch Iteration')
      plt.ylabel('Mean Squared Error (MSE)')
      plt.grid(True, linestyle='--', alpha=0.6)
      plt.legend()

      # Adding background for better readability
      plt.tight_layout()
      plt.show()

    plot_loss(history_loss)

    def plot_training_results(losses, bces, uncs):
      fig, ax1 = plt.subplots(figsize=(10, 6))

      # Primary axis for Loss
      ax1.plot(losses, alpha=0.2, color='royalblue', label='Total Loss')
      # Smoothed BCE trend
      if len(bces) > 5:
          smooth_bce = [sum(bces[max(0, i-10):i+1]) / len(bces[max(0, i-10):i+1]) for i in range(len(bces))]
          ax1.plot(smooth_bce, color='red', linewidth=2, label='BCE Trend (Accuracy Indicator)')

      ax1.set_xlabel('Batch Iteration')
      ax1.set_ylabel('Loss / BCE', color='red')
      ax1.tick_params(axis='y', labelcolor='red')
      ax1.grid(True, linestyle='--', alpha=0.5)

      # Secondary axis for Uncertainty
      ax2 = ax1.twinx()
      if len(uncs) > 5:
          smooth_unc = [sum(uncs[max(0, i-10):i+1]) / len(uncs[max(0, i-10):i+1]) for i in range(len(uncs))]
          ax2.plot(smooth_unc, color='green', linestyle=':', label='Uncertainty Trend')

      ax2.set_ylabel('Uncertainty (U-L)', color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      plt.title('JLNN Training: PointNet Learning vs Logic Constraints')
      fig.tight_layout()
      ax1.legend(loc='upper left')
      ax2.legend(loc='upper right')
      plt.show()

    plot_training_results(history_loss, history_bce, history_unc)


Conclusion
-------------

By combining JAX's high-performance autodiff with JLNN's symbolic grounding, we created a 3D classifier that is not only accurate but also explainable. This approach allows developers to debug *why* a model misclassified an object by looking at individual predicate activations.

Download
-----------

You can also download the raw notebook file for local use:
:download:`JLNN_pointcloud_classification.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_pointcloud_classification.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.