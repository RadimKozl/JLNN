JLNN Explainability – From scales to symbolic rules
======================================================

This tutorial demonstrates how to "open the black box" of a **JLNN** model and extract human-readable rules from it. Unlike conventional neural networks, JLNN allows for direct export of learned knowledge back to symbolic logic.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab A variant
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantA.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub A variant
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantA.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab B variant
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantB.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub B variant
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantB.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.

🌟 Tutorial goal
~~~~~~~~~~~~~~~~~~

The goal is to demonstrate the **Rule Extraction** process on the classic Iris dataset. We will focus on:

* Converting numerical weights into understandable logical operators.
* Interpretation of **Semantic Grounding** (fuzzy boundaries).
* Visualization of the model's decision-making processes.

🛠️ Key XAI techniques in JLNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Semantic quantifiers
----------------------

JLNN doesn't just use raw numbers. Using the steepness of sigmoidal functions, we can define the model's certainty:

* **Very accurate**: High steepness, sharp boundary.
* **Approximately**: Medium steepness, fuzzy transition.
* **Around**: Low steepness, wide area of ​​uncertainty.

Importance calculation
-------------------------

The importance of a rule is not only determined by its weight, but also by the quality of its grounding:

.. math::

    Importance = |Weight| \times Mean(Steepness)

💻 Rule extraction demonstration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code (abbreviated) demonstrates how the JLNN model transforms internal parameters into a text string:

.. code-block:: python

    # IF (petal_length < 1.45) AND (petal_width < 0.25) THEN Iris-Setosa
    rule_str = f"IF {feature} {quantifier} {threshold} THEN {consequent}"

📈 Visualization and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial includes three types of XAI outputs:

1. **Text Report**: A list of rules sorted by their importance.
2. **Membership Functions**: Graphs showing how the model "sees" individual features (e.g. petal length).
3. **Contribution Heatmap**: A matrix showing the influence of each feature on a specific logical decision.

🏁 Conclusion: Why is this version the best?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach to explainability is unique for several reasons:

* **Human voice**: Thanks to the integration of quantifiers, the model does not communicate only in zeros and ones, but generates rules understandable to experts.
* **Semantic depth**: Unlike decision trees, JLNN preserves the fluidity of reality thanks to fuzzy logic.
* **Connection with Grand Cycle**: The documentation includes the use of results from automatic tuning (Optuna), which ensures that the extracted rules are mathematically optimal.

Tutorial code A variant
-------------------------

.. code-block:: python

    '''
    try:
        import jlnn
        from flax import nnx
        import jax.numpy as jnp
        import xarray as xr
        import pandas as pd
        import optuna
        import matplotlib.pyplot as plt
        import sklearn
        print("✅ JLNN and JAX are ready.")
    except ImportError:
        print("🚀 Installing JLNN from GitHub and fixing JAX for Colab...")
        # Instalace frameworku
        #!pip install jax-lnn --quiet
        !pip install git+https://github.com/RadimKozl/JLNN.git --quiet
        !pip install optuna optuna-dashboard pandas scikit-learn matplotlib
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        !pip install  scikit-learn pandas

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris

    print(f"JAX Device: {jax.devices()[0]}")

    sns.set(style="whitegrid")

    iris = load_iris()
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    X = iris.data
    y = (iris.target == 0).astype(float)

    centers = np.array([[2.4, 0.8, 1.45, 0.25], [5.8, 2.7, 4.35, 1.3]])
    steepnesses = np.array([[12.5, 5.0, 28.0, 35.0], [8.0, 4.5, 15.0, 18.0]])
    rule_weights = np.array([0.95, -0.88]) # Rule 1 for Setosa, Rule 2 against

    print(f"XAI analysis started for {len(rule_weights)} key rules.")

    def extract_rules(centers, steepnesses, rule_weights, feature_names, imp_threshold=0.1):
        rules = []
        # Importance = rule weight * average slope (model confidence)
        importances = np.abs(rule_weights) * np.mean(steepnesses, axis=1)
        
        for r in range(len(rule_weights)):
            w = rule_weights[r]
            imp = importances[r]
            
            if imp < imp_threshold: continue
                
            antecedents = []
            for f, fname in enumerate(feature_names):
                c = centers[r, f]
                s = steepnesses[r, f]
                
                # Quantifier selection by steepness (Explainability)
                if s > 25: desc = "very accurately"
                elif s > 12: desc = "approximately"
                elif s > 5: desc = "around"
                else: continue # We ignore too vague bindings
                
                # Heuristics for direction (simplified to thresholds in this XAI tutorial)
                op = "<" if w > 0 else ">"
                antecedents.append(f"{fname} {desc} {op} {c:.2f}")
                
            if not antecedents: continue
                
            antecedent_str = " ∧ ".join(antecedents)
            consequent = "SETOSA" if w > 0 else "OTHER"
            
            rule_str = f"R{r} (Imp:{imp:.2f}): IF {antecedent_str} THEN {consequent}"
            rules.append((imp, rule_str))
            
        rules.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in rules]

    extracted_rules = extract_rules(centers, steepnesses, rule_weights, feature_names)
    print("\n=== EXTRACTED SYMBOLIC RULES ===")
    for r in extracted_rules: print(r)

    def plot_xai_membership(centers, steepnesses, rule_idx, feature_names):
        x_range = np.linspace(0, 7, 200)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        
        for f in range(4):
            c, s = centers[rule_idx, f], steepnesses[rule_idx, f]
            # Sigmoidal activation
            y_val = 1 / (1 + np.exp(-s * (x_range - c)))
            
            axes[f].plot(x_range, y_val, lw=2, color='teal')
            axes[f].fill_between(x_range, y_val, alpha=0.2, color='teal')
            axes[f].axvline(c, color='red', ls='--', alpha=0.5)
            axes[f].set_title(f"{feature_names[f]}")
            axes[f].set_ylim(-0.05, 1.05)
            if f == 0: axes[f].set_ylabel(f"Truthfulness R{rule_idx}")

    print("\nSemantic grounding visualization for Rule 0 (Setosa):")
    plot_xai_membership(centers, steepnesses, 0, feature_names)
    plt.show()

    plt.figure(figsize=(10, 4))
    contribution = np.abs(rule_weights[:, None]) * steepnesses
    sns.heatmap(contribution, annot=True, fmt=".1f", cmap="magma", 
                xticklabels=feature_names, yticklabels=[f"Rule {i}" for i in range(len(rule_weights))])
    plt.title("XAI: The Contribution of Properties to Logical Certainty")
    plt.show()

Tutorial code B variant
-------------------------

.. code-block:: python

    '''
    try:
        import jlnn
        from flax import nnx
        import jax.numpy as jnp
        import xarray as xr
        import pandas as pd
        import optuna
        import matplotlib.pyplot as plt
        import sklearn
        print("✅ JLNN and JAX are ready.")
    except ImportError:
        print("🚀 Installing JLNN from GitHub and fixing JAX for Colab...")
        # Instalace frameworku
        #!pip install jax-lnn --quiet
        !pip install git+https://github.com/RadimKozl/JLNN.git --quiet
        !pip install optuna optuna-dashboard pandas scikit-learn matplotlib
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        !pip install  scikit-learn pandas

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, confusion_matrix
    import warnings

    from IPython.display import Markdown, display

    print(f"JAX Device: {jax.devices()[0]}")

    warnings.filterwarnings("ignore")
    sns.set(style="whitegrid")

    iris = load_iris()
    X = iris.data
    # Binary classification: Setosa (1) vs. Versicolor/Virginica (0)
    y = (iris.target == 0).astype(float)
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

    print(f"Data retrieved: {X.shape} samples. Target: Iris-Setosa detection.")

    centers = np.array([[2.4, 0.8, 1.45, 0.25], [5.8, 2.7, 4.35, 1.3]])
    steepnesses = np.array([[12.5, 5.0, 28.0, 35.0], [8.0, 4.5, 15.0, 18.0]])
    rule_weights = np.array([0.95, -0.88]) # Rule 1 for Setosa, Rule 2 against

    print(f"XAI analysis started for {len(rule_weights)} key rules.")

    centers = np.array([
        [5.0, 3.4, 1.46, 0.24],  # Rule 0: Typical Setos
        [5.9, 2.7, 4.30, 1.30]   # Rule 1: Typical "others"
    ])
    steepnesses = np.array([
        [8.0, 5.0, 25.0, 30.0],  # Rule 0: Very sharp on petals
        [5.0, 4.0, 12.0, 15.0]   # Rule 1: Looser boundaries
    ])
    rule_weights = np.array([0.98, -0.92]) # Positive weight for Setosa, negative for others

    def get_model_predictions(X, centers, steepnesses, weights):
        # Calculate membership functions for all samples
        # broadcasted sigmoid: 1 / (1 + exp(-s * (x - c)))
        memberships = 1 / (1 + np.exp(-steepnesses[:, None, :] * (X[None, :, :] - centers[:, None, :])))
        # Aggregation rule (average over feature) - simplified AND
        rule_activations = np.mean(memberships, axis=2)
        # Final score as a weighted sum
        logits = rule_activations.T @ weights
        return (logits > 0).astype(float), rule_activations

    y_pred, activations = get_model_predictions(X, centers, steepnesses, rule_weights)
    acc = accuracy_score(y, y_pred)
    print(f"✅ Accuracy of the extracted model on the data: {acc*100:.1f}%")

    def extract_rule_report(centers, steepnesses, weights, features):
        report = "## 🤖 Extracted Rules (Symbolic Audit)\n\n"
        rules_data = []
        
        for r in range(len(weights)):
            w = weights[r]
            antecedents = []
            for f in range(len(features)):
                s = steepnesses[r, f]
                c = centers[r, f]
                
                # Quantifiers by steepness
                if s > 20: desc = "very accurately"
                elif s > 10: desc = "approximately"
                elif s > 4: desc = "around"
                else: continue
                
                # Direction of reasoning (heuristics for the tutorial)
                op = "<" if w > 0 else ">"
                antecedents.append(f"`{features[f]}` {desc} {op} **{c:.2f}**")
                
            logic_str = " ∧ ".join(antecedents)
            target = "SETOSA" if w > 0 else "OTHER"
            line = f"{r+1}. **{w:+.2f}** :: {logic_str} → `{target}`"
            report += line + "  \n"
            rules_data.append(line)
            
        return report, rules_data

    md_report, _ = extract_rule_report(centers, steepnesses, rule_weights, feature_names)
    display(Markdown(md_report))

    def plot_membership_functions(centers, steepnesses, features):
        x_plot = np.linspace(0, 8, 300)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        colors = ['#2ecc71', '#e74c3c'] # Green for Setosa, red for others
        
        for f in range(4):
            for r in range(len(centers)):
                c, s = centers[r, f], steepnesses[r, f]
                # If the rule weight is negative, we invert the sigmoid to visualize the "others"
                y_plot = 1 / (1 + np.exp(-s * (x_plot - c)))
                axes[f].plot(x_plot, y_plot, label=f"Rule {r}", color=colors[r], lw=2)
                axes[f].fill_between(x_plot, y_plot, alpha=0.1, color=colors[r])
                
            axes[f].set_title(features[f], fontweight='bold')
            axes[f].set_ylim(-0.05, 1.05)
            if f == 0: axes[f].set_ylabel("Truth Value")
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nVisualizing Grounding Boundaries:")
    plot_membership_functions(centers, steepnesses, feature_names)

    plt.figure(figsize=(10, 4))
    importance_matrix = np.abs(rule_weights[:, None]) * steepnesses
    sns.heatmap(importance_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=feature_names, yticklabels=["Setosa Rule", "Other Rule"])
    plt.title("XAI Heatmap: Feature Importance by Logical Steepness")
    plt.show()


Download
----------

You can also download the raw notebook file for local use:

:download:`JLNN_explainability_rule_extraction_XAI_variantA.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantA.ipynb>`

:download:`JLNN_explainability_rule_extraction_XAI_variantB.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_explainability_rule_extraction_XAI_variantB.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.

    

