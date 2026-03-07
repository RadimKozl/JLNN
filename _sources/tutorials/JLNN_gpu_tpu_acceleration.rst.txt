JLNN – Accelerated Interval Logic
==========================================

Massive toxicity screening on the ToxCast dataset (GPU & TPU Edition)
-------------------------------------------------------------------------------

In this tutorial, we will show how **Justifiable Logical Neural Networks (JLNN)** natively handle uncertainty using **truth intervals**. Unlike standard neural networks that provide a single point estimate, JLNN provides a range $[L, U]$ that represents both the probability and the confidence of the prediction.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_gpu_tpu_acceleration.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_gpu_tpu_acceleration.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.



Theoretical Background: Why Interval Logic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard Neural Networks (NNs) usually provide a single point estimate (e.g., $P(toxic) = 0.85$). However, in high-stakes fields like toxicology, we need to distinguish between **risk** (probability) and **uncertainty** (lack of evidence).

**Truth Intervals** $[L, U]$ represent:
* **[1.0, 1.0]**: Absolute Truth.
* **[0.0, 0.0]**: Absolute Falsehood.
* **[0.0, 1.0]**: Complete Uncertainty (Lack of data).
* **[0.7, 0.8]**: High probability with low uncertainty.

The Logical Formula
----------------------

We are training the model to respect the following logical rule:

.. math::

   0.75 :: (AhR\_active \land DNA\_damage) \implies HighToxicity

This means that if a molecule activates the AhR receptor **AND** causes DNA damage, it is at least 75% likely to be highly toxic.

Example
---------

.. code-block:: python

   import os
   import sys

   try:
      # Checking if everything is ready (after reboot)
      import jax
      import jlnn
      import deepchem as dc
      from flax import nnx
      import jax.numpy as jnp

      # Initialization confirmation
      backend = jax.default_backend()
      print(f"✅ JLNN and JAX are ready. Running on: {backend.upper()}")
      print(f"🔢 Devices: {jax.devices()}")

   except (ImportError, RuntimeError):
      print("🚀 Initializing environment (Installing JLNN and fixing JAX)...")

      # A. Hardware detection
      is_tpu = 'TPU_NAME' in os.environ
      is_gpu = False
      try:
         import subprocess
         subprocess.check_output('nvidia-smi')
         is_gpu = True
      except:
         pass

      # B. Install JAX (MUST BE FIRST to avoid CPU-only version from JLNN)
      if is_tpu:
         print("⚡ TPU detected. Installing jax[tpu]...")
         !pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      elif is_gpu:
         print("🔥 GPU detected. Installing jax[cuda12_pip]...")
         !pip install -q --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
      else:
         print("💻 No accelerator found. Using jax[cpu].")
         !pip install -q --upgrade "jax[cpu]"

      # C. Installing JLNN and scientific libraries
      print("📦 Installing JLNN framework and chemical dependencies...")
      !pip install -q git+https://github.com/RadimKozl/JLNN.git --quiet
      !pip install -q deepchem rdkit jraph numpyro optuna pandas scikit-learn matplotlib xarray arviz --quiet

      # D. RESTART KERNEL (Necessary to load new drivers in Colab)
      print("\n🔄 RESTARTING ENVIRONMENT... Run this cell again after the restart.")
      os.kill(os.getpid(), 9)

   # Imports

   import os
   import time
   import warnings
   import jax
   import jax.numpy as jnp
   import optax
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from flax import nnx
   from rdkit import Chem
   from rdkit import RDLogger
   from rdkit.Chem import AllChem
   from jlnn.symbolic.compiler import LNNFormula
   from jlnn.training.losses import contradiction_loss
   from rdkit.Chem import rdFingerprintGenerator

   # Device Check
   backend = jax.default_backend()
   print(f"✅ JLNN initialized to: {backend.upper()}")
   print(f"🔢 Available devices: {jax.devices()}")

   # --- LOADING REAL DATA (TOXCAST) ---

   print("\n📥 Loading ToxCast dataset...")
   url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
   df = pd.read_csv(url, compression='gzip').head(2000)

   print(f"🧪 Processing {len(df)} molecules using RDKit...")

   def smiles_to_fp(smiles):
      try:
         mol = Chem.MolFromSmiles(smiles)
         if mol:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            fp = gen.GetFingerprintAsNumPy(mol)
            return np.array(fp, dtype=np.float32)
      except:
         return None
      return None

   # Disables RDKit warnings in the log
   RDLogger.DisableLog('rdApp.*')
   # Disables standard Python warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)

   df_subset = df.head(2000).copy()
   fps = [smiles_to_fp(s) for s in df_subset['smiles']]

   # Filtering valid molecules

   valid_mask = [f is not None for f in fps]
   X_train = jnp.array([f for f in fps if f is not None])

   y_column = df.columns[1]
   Y_train = jnp.array(df_subset[y_column].values[valid_mask], dtype=jnp.float32).reshape(-1, 1)

   # Removing NaN in the target variable

   nan_mask = ~jnp.isnan(Y_train).flatten()
   X_train, Y_train = X_train[nan_mask], Y_train[nan_mask]

   print(f"✅ Data ready for JLNN!")
   print(f"   Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
   print(f"   Target assay: {y_column}")

   # Massive screening simulation (100k molecules)

   X_massive = jnp.tile(X_train[:1000], (100, 1))

   # --- DEFINITION OF A PURE JLNN MODEL (INTERVAL LOGIC) ---
   # We define an expert rule: If AhR and p53 are activated, there is a risk of high toxicity

   formula_str = "0.75::(AhR_active & DNA_damage) -> HighToxicity"
   formula = LNNFormula(formula_str, nnx.Rngs(123))

   # Grounding: Convert fingerprints to fuzzy intervals [Lower, Upper]
   def get_grounding(X):
      # For the purposes of the tutorial, we map parts of the fingerprint to logical literals
      # In a real application, there would be a trainable layer LearnedPredicate
      ahr_val = jax.nn.sigmoid(X[:, :512].mean(axis=1))
      dna_val = jax.nn.sigmoid(X[:, 512:].mean(axis=1))

      # Create intervals: [center - epsilon, center + epsilon]
      epsilon = 0.05
      return {
         "AhR_active": jnp.stack([jnp.clip(ahr_val - epsilon, 0, 1), jnp.clip(ahr_val + epsilon, 0, 1)], axis=-1),
         "DNA_damage": jnp.stack([jnp.clip(dna_val - epsilon, 0, 1), jnp.clip(dna_val + epsilon, 0, 1)], axis=-1)
      }

   # --- TRAINING STEP (LOGIC OPTIMIZATION) ---

   # Optimizer is defined here; opt_state will be initialized after formula recreation in the next cells
   optimizer = optax.adam(0.01)

   @nnx.jit
   def train_step(formula, opt_state, X, y):
      def loss_fn(f):
         # Grounding with fixed 'HighToxicity'
         grounding = get_grounding(X)
         truth_intervals = f(grounding)

         # MSE Loss (interval midpoint vs label)
         mid_point = (truth_intervals[..., 0] + truth_intervals[..., 1]) / 2
         mse_loss = jnp.mean((mid_point - y)**2)

         # Contradiction Loss (guards the integrity of L <= U)
         c_loss = contradiction_loss(truth_intervals).mean()

         # 🔥 KEY FIX: Force all keys in gradient
         # Add a "virtual zero" to all parameters so that JAX doesn't skip the 'right' branch
         all_params = nnx.state(f, nnx.Param)
         param_sum = sum(jnp.sum(p) for p in jax.tree.leaves(all_params) if p is not None)

         return mse_loss + 0.2 * c_loss + 0.0 * param_sum, truth_intervals

      # Calculation of gradients
      (loss, intervals), grads = nnx.value_and_grad(loss_fn, has_aux=True)(formula)

      # Update via nnx.state
      current_params = nnx.state(formula, nnx.Param)
      updates, opt_state = optimizer.update(grads, opt_state, current_params)
      nnx.update(formula, updates)

      return loss, opt_state, intervals

   # Clean initialization - I recommend starting this cell all over again
   formula = LNNFormula("0.75::(AhR_active & DNA_damage) -> HighToxicity", nnx.Rngs(123))
   optimizer = optax.adam(0.01)
   opt_state = optimizer.init(nnx.state(formula, nnx.Param))

   # We reset the state before the loop to be sure
   params_state = nnx.state(formula, nnx.Param)
   opt_state = optimizer.init(params_state)

   # Training loop

   def get_grounding(X):
      # 1. Inputs (Antecedent)
      ahr_val = jax.nn.sigmoid(X[:, :512].mean(axis=1))
      dna_val = jax.nn.sigmoid(X[:, 512:].mean(axis=1))
      eps = 0.05
      batch_size = X.shape[0]

      # 2. Goal (Consequent) - HighToxicity
      # We need to define a starting interval [0, 1] for all samples
      high_tox_init = jnp.tile(jnp.array([0.0, 1.0]), (batch_size, 1))

      return {
         "AhR_active": jnp.stack([jnp.clip(ahr_val-eps, 0, 1), jnp.clip(ahr_val+eps, 0, 1)], axis=-1),
         "DNA_damage": jnp.stack([jnp.clip(dna_val-eps, 0, 1), jnp.clip(dna_val+eps, 0, 1)], axis=-1),
         "HighToxicity": high_tox_init
      }

   # Starting the loop

   losses = []
   print("🧠 I'm practicing JLNN interval logic...")

   for i in range(250):
      try:
         loss, opt_state, final_intervals = train_step(formula, opt_state, X_train[:1000], Y_train[:1000])
         losses.append(float(loss))

         if i % 50 == 0:
               print(f"  Iterace {i:3}: Loss = {loss:.4f}")
      except Exception as e:
         print(f"❌ Error in iteration {i}: {e}")
         break

   print("✅ Training completed.")

   # --- HARDWARE BENCHMARK (MASSIVE SCREENING) ---

   print(f"\n🧪 Running screening on {backend.upper()}...")
   grounding_massive = get_grounding(X_massive)

   # TPU optimization (bfloat16)
   if backend == 'tpu':
      grounding_massive = {k: v.astype(jnp.bfloat16) for k, v in grounding_massive.items()}

   start = time.time()
   # JIT compiled inference
   final_intervals = jax.jit(lambda g: formula(g))(grounding_massive)
   final_intervals.block_until_ready()
   duration = time.time() - start
   print(f" ✅ Screening done in {duration:.4f} s")

   # --- VISUALIZATION WITH DIMENSION CORRECTION ---

   fig, ax = plt.subplots(1, 2, figsize=(16, 6))

   # Graph A: Loss (remains the same)
   ax[0].plot(losses, color='#2ca02c', lw=2)
   ax[0].set_title("JLNN Learning Process")
   ax[0].set_xlabel("Iterace")
   ax[0].set_ylabel("Loss")

   # Graph B: Confidence intervals
   # final_intervals has the form (batch, nodes, 2)
   # We only want the first 60 molecules and ONLY the last node (index -1)
   prediction_intervals = final_intervals[:60]

   # If prediction_intervals is (60, 2, 2), we need to take the right node
   # Usually it is the last node in the graph:
   L = prediction_intervals[:, -1, 0]
   U = prediction_intervals[:, -1, 1]

   # Check for Matplotlib (must be 60)
   idx = jnp.arange(L.shape[0])

   ax[1].fill_between(idx, L, U, color='royalblue', alpha=0.3, label='Truth interval [L, U]')
   ax[1].plot(idx, (L+U)/2, 'o-', markersize=3, color='darkblue', label='Mean value')

   ax[1].set_title(f"Logical screening output (60 samples)")
   ax[1].set_xlabel("Molecule Index")
   ax[1].set_ylabel("Toxicity level")
   ax[1].set_ylim(-0.05, 1.05)
   ax[1].legend()

   plt.tight_layout()
   plt.show()

Conclusion
~~~~~~~~~~~~~

JLNN provides a transparent way to handle biological screening. Unlike "black-box" models, we can see exactly where the model is uncertain, allowing scientists to prioritize molecules that require further experimental validation.

Download
----------

You can also download the raw notebook file for local use:
:download:`JLNN_gpu_tpu_acceleration.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_gpu_tpu_acceleration.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.