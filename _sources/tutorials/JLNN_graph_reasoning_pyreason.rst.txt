Differentiable Reasoning on Graphs (JLNN vs. PyReason)
========================================================

This tutorial demonstrates how to use **JLNN** (Logical Neural Networks in JAX) for logical reasoning over graph data as a modern, educational alternative to tools like PyReason.

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_graph_reasoning_pyreason.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_graph_reasoning_pyreason.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.


🌟 Key concepts
~~~~~~~~~~~~~~~~~

While traditional systems (e.g. PyReason) work with fixed rules and fixed thresholds, JLNN delivers:

* **Rule Weight Learning**: The model automatically optimizes the weight of "social influence" versus "own attributes" based on the data.
* **Semantic Grounding**: Allows learning of fuzzy boundaries using trainable parameters (e.g. a sigmoidal function determining what exactly defines a "cool car").
* **End-to-end training**: The entire chain of reasoning is fully differentiable through graph operations thanks to the JAX and Flax frameworks.

🛠️ Model architecture
~~~~~~~~~~~~~~~~~~~~~~~

The model solves the spread of "popularity" in a social network using two chained rules:

1. Local rule (Node attributes)
---------------------------------

It defines the local trendiness of a node based on its direct properties:

.. math::

   0.92 :: (has\_cool\_pet \land has\_cool\_car) \to is\_trendy\_local

2. Social Rule (Graphic Reasoning)
------------------------------------

It defines the resulting social popularity by combining the influence of neighbors and local status:

.. math::

   0.85 :: (is\_friend \land is\_trendy\_local) \to is\_trendy\_social

💻 Implementation details
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with intervals in JAX
-------------------------------

JLNN represents truth as intervals :math:`[L, U]` (Lower, Upper bound).

.. code-block:: python

    # Propagation of truth intervals over the adjacency matrix (adj)
    # Result is the average truth around the node
    friend_influence = jnp.matmul(adj, local_trendy_interval) / jnp.sum(adj, axis=1, keepdims=True)

Symbol initialization
-----------------------

When calling ``LNNFormula``, the input dictionary ``inputs`` must contain all symbols. For predicates that are results (consequents), we initialize the state "unknown" :math:`[0, 1]`:

.. code-block:: python

    inputs = {
        "has_cool_pet": pet_data,    # Pozorovaná data (vstupy)
        "is_trendy_local": unknown,  # Cíl výpočtu (inicializováno na [0.0, 1.0])
    }

📈 Training and results
~~~~~~~~~~~~~~~~~~~~~~~~~

The model uses ``total_lnn_loss``, which penalizes:

1. **Logical contradictions**: For example, a situation where the lower limit exceeds the upper limit (:math:`L > U`).
2. **Prediction Error**: Euclidean distance between the predicted interval and the target value.

**Tutorial Outputs:**

* **Visual graph map**: Color coding of nodes corresponding to learned logical truth.
* **Optimized Parameters**: The model after training contains specific learned thresholds for interpreting the input data.

Example
---------

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
    import optax
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.training.losses import total_lnn_loss

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep output clean

    from sklearn.metrics import accuracy_score

    print(f"JAX Device: {jax.devices()[0]}")

    G = nx.Graph()

    people = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank"]
    G.add_nodes_from(people)

    friendships = [
        ("Alice", "Bob"), ("Alice", "Charlie"), ("Bob", "Dana"),
        ("Charlie", "Eve"), ("Dana", "Frank"), ("Eve", "Grace"),
        ("Frank", "Hank"), ("Grace", "Hank"), ("Bob", "Eve")
    ]
    G.add_edges_from(friendships)
    node_list = list(G.nodes())

    pet_scores_dict = {"Alice": 0.8, "Bob": 0.3, "Charlie": 0.9, "Dana": 0.4,
                   "Eve": 0.7,   "Frank": 0.2, "Grace": 0.85, "Hank": 0.6}

    car_scores_dict = {"Alice": 0.6, "Bob": 0.9, "Charlie": 0.4, "Dana": 0.8,
                   "Eve": 0.5,   "Frank": 0.95, "Grace": 0.7, "Hank": 0.3}

    nx.set_node_attributes(G, pet_scores_dict, "pet_score")
    nx.set_node_attributes(G, car_scores_dict, "car_score")

    pet_scores = jnp.array([pet_scores_dict[n] for n in node_list])
    car_scores = jnp.array([car_scores_dict[n] for n in node_list])
    adj_matrix = jnp.array(nx.to_numpy_array(G))

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_weight="bold")
    plt.title("Social Network: Friends + Ownership")
    plt.show()

    class TrainableFuzzy(nnx.Module):
        def __init__(self, name: str, init_center: float = 0.5, init_steep: float = 10.0):
            self.name = name
            self.center = nnx.Param(jnp.array([init_center]))
            self.steep  = nnx.Param(jnp.array([init_steep]))

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return 1.0 / (1.0 + jnp.exp(-jnp.abs(self.steep) * (x - self.center)))

    degree     = dict(G.degree())
    max_degree = max(degree.values())
    popularity = {
        n: degree[n] / max_degree
        + 0.3 * pet_scores_dict[n]
        + 0.2 * car_scores_dict[n]
        for n in G.nodes()
    }
    popularity = {n: float(np.clip(v, 0.0, 1.0)) for n, v in popularity.items()}

    print("\n" + "="*60)
    print("EXPERIMENT A: Single-rule prototype")
    print("="*60)

    rule_A = "0.92 :: (has_cool_pet & has_cool_car) -> is_trendy"
    logic_A = LNNFormula(rule_A, rngs=nnx.Rngs(42))

    class GraphLNN_A(nnx.Module):
        """Single-rule LNN with fuzzy grounding.

        Rule: (has_cool_pet & has_cool_car) -> is_trendy

        LNNFormula always expects a dict[str, jnp.ndarray] where each value has
        shape (batch, 2) representing a truth interval [lower, upper].
        Passing a raw tensor causes a JAX string-indexing TypeError.
        """

        def __init__(self, logic_model):
            self.logic = logic_model
            self.c_pet  = nnx.Param(jnp.array([0.5]))
            self.c_car  = nnx.Param(jnp.array([0.5]))
            self.steep  = nnx.Param(jnp.array([10.0]))

        def __call__(self, pet_val, car_val, adj):
            n = pet_val.shape[0]

            # Fuzzy grounding → scalar membership per node
            pet_f = 1.0 / (1.0 + jnp.exp(-jnp.abs(self.steep) * (pet_val - self.c_pet)))
            car_f = 1.0 / (1.0 + jnp.exp(-jnp.abs(self.steep) * (car_val - self.c_car)))

            # Wrap scalars into [lower, upper] interval tensors of shape (n, 2).
            # A small epsilon gap keeps the interval non-degenerate.
            pet_b = jnp.stack([pet_f, pet_f + 0.01], axis=-1)   # (n, 2)
            car_b = jnp.stack([car_f, car_f + 0.01], axis=-1)   # (n, 2)

            # Maximally uncertain interval for the consequent prior
            unknown = jnp.tile(jnp.array([0.0, 1.0]), (n, 1))   # (n, 2)

            # FIX: pass a dict with string keys – LNNFormula looks up literals by name.
            # friend_f is NOT included here; this rule only has two antecedent literals.
            # Graph diffusion is handled in Experiment B via the two-rule architecture.
            inputs = {
                "has_cool_pet": pet_b,
                "has_cool_car": car_b,
                "is_trendy":    unknown,
            }
            return self.logic(inputs)   # (n, 2)

    graph_model_A = GraphLNN_A(logic_A)
    optimizer_A   = nnx.Optimizer(graph_model_A, optax.adamw(0.01), wrt=nnx.Param)

    targets_A = jnp.stack(
        [pet_scores * 0.5 + car_scores * 0.5,
        pet_scores * 0.5 + car_scores * 0.5 + 0.05],
        axis=1,
    )

    @nnx.jit
    def train_step_A(m, opt, p_data, c_data, adj, targets):
        def loss_fn(model_ptr):
            # LNNFormula returns (n_nodes, n_literals, 2):
            #   axis-1 index 0 = antecedent (has_cool_pet & has_cool_car)
            #   axis-1 index 1 = consequent (is_trendy)  ← this is what we train against
            preds = model_ptr(p_data, c_data, adj)   # (8, 2, 2)
            return total_lnn_loss(preds[:, 1, :], targets, contradiction_weight=2.0)
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    print("Training Experiment A...")
    for step in range(1001):
        loss_A = train_step_A(graph_model_A, optimizer_A,
                            pet_scores, car_scores, adj_matrix, targets_A)
        if step % 250 == 0:
            print(f"  Step {step:4d} | Loss: {loss_A:.6f}")

    print("\n" + "="*60)
    print("EXPERIMENT B: Two-rule graph diffusion")
    print("="*60)

    class GraphLNN_B(nnx.Module):
        """Two-rule LNN: local trendiness propagated through the friendship graph."""

        def __init__(self):
            self.rule_local  = LNNFormula(
                "0.92 :: (has_cool_pet & has_cool_car) -> is_trendy_local",  nnx.Rngs(42))
            self.rule_social = LNNFormula(
                "0.85 :: (is_friend & is_trendy_local) -> is_trendy_social", nnx.Rngs(43))

            self.c_pet  = nnx.Param(jnp.array([0.5]))
            self.c_car  = nnx.Param(jnp.array([0.5]))
            self.steep  = nnx.Param(jnp.array([12.0]))

        def __call__(self, pet_val, car_val, adj):
            batch_size = pet_val.shape[0]
            # Maximally uncertain interval [0, 1] – used as consequent prior.
            unknown = jnp.tile(jnp.array([0.0, 1.0]), (batch_size, 1))   # (n, 2)

            # 1. Fuzzy grounding → interval beliefs
            pet_f  = 1.0 / (1.0 + jnp.exp(-jnp.abs(self.steep) * (pet_val - self.c_pet)))
            car_f  = 1.0 / (1.0 + jnp.exp(-jnp.abs(self.steep) * (car_val - self.c_car)))
            pet_b  = jnp.stack([pet_f,  pet_f  + 0.01], axis=-1)   # (n, 2)
            car_b  = jnp.stack([car_f,  car_f  + 0.01], axis=-1)   # (n, 2)

            # 2. Rule 1 – local trendiness
            inputs1 = {
                "has_cool_pet":   pet_b,
                "has_cool_car":   car_b,
                "is_trendy_local": unknown,
            }
            local_trendy_full = self.rule_local(inputs1)     # (n, 3, 2): [antecedent_and, pet, car, consequent]
            local_trendy = local_trendy_full[:, 2, :]          # (n, 2): consequent is_trendy_local only

            # 3. Graph diffusion – average neighbour belief
            # adj: (n, n), local_trendy: (n, 2) → friend_sum: (n, 2)
            friend_sum   = jnp.matmul(adj, local_trendy)
            friend_count = jnp.sum(adj, axis=1, keepdims=True)          # (n, 1)
            friend_b     = friend_sum / jnp.where(friend_count > 0, friend_count, 1.0)

            # 4. Rule 2 – social trendiness
            inputs2 = {
                "is_friend":        friend_b,
                "is_trendy_local":  local_trendy,
                "is_trendy_social": unknown,
            }
            return self.rule_social(inputs2)   # (n, 2)

    model_B     = GraphLNN_B()
    optimizer_B = nnx.Optimizer(model_B, optax.adamw(0.02), wrt=nnx.Param)

    degree_vec   = jnp.sum(adj_matrix, axis=1)                        # (n,)
    safe_degree  = jnp.where(degree_vec > 0, degree_vec, 1.0)
    neigh_pet    = jnp.matmul(adj_matrix, pet_scores) / safe_degree   # mean neighbour pet score

    target_val      = pet_scores * 0.4 + car_scores * 0.2 + neigh_pet * 0.4
    target_val      = jnp.clip(target_val, 0.0, 1.0)                  # FIX: clip to valid range
    target_interval = jnp.stack([target_val, target_val + 0.05], axis=1)

    @nnx.jit
    def train_step_B(m, opt, p_data, c_data, adj, targets):
        def loss_fn(model_ptr):
            # rule_social has 3 literals: is_friend, is_trendy_local, is_trendy_social
            # LNNFormula output shape: (n_nodes, n_literals, 2) = (8, 3, 2)
            #   index 0 = is_friend
            #   index 1 = is_trendy_local
            #   index 2 = is_trendy_social  ← consequent, train against this
            preds = model_ptr(p_data, c_data, adj)   # (8, 3, 2)
            return total_lnn_loss(preds[:, 2, :], targets, contradiction_weight=3.0)
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    print("Training Experiment B...")
    for step in range(1201):
        loss_B = train_step_B(model_B, optimizer_B,
                            pet_scores, car_scores, adj_matrix, target_interval)
        if step % 300 == 0:
            print(f"  Step {step:4d} | Loss: {loss_B:.6f}")

    final_preds  = model_B(pet_scores, car_scores, adj_matrix)  # (8, 3, 2)
    trendy_lower = np.array(final_preds[:, 2, 0])               # consequent lower bound → colour

    plt.figure(figsize=(8, 6))
    pos   = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=trendy_lower,
                                cmap="viridis", node_size=800)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Final Learned Trendiness (lower bound)")
    plt.colorbar(nodes)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("EXPERIMENT C: Optuna hyperparameter search")
    print("="*60)

    popularity_gt = jnp.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.6, 0.85, 0.55])
    degrees_c = jnp.sum(adj_matrix, axis=1) + 1e-6   # avoid div-by-zero

    def fuzzy_high(x, center, steepness):
        """Sigmoid membership: 1 when x >> center, 0 when x << center."""
        return 1.0 / (1.0 + jnp.exp(-steepness * (x - center)))

    def aggregate_friends(popularity, adj, deg):
        """Weighted-average popularity of graph neighbours."""
        return jnp.matmul(adj, popularity) / deg

    def model_forward_c(params, car, pet, adj, deg):
        """
        Returns a [0,1] popularity score for each of the 8 nodes.

        Fuzzy rule: high_car AND high_pet  → direct trendiness
        Propagation: blend direct score with neighbourhood average (2 steps).
        """
        high_car = fuzzy_high(car, params["c_car"],  params["steepness"])
        high_pet = fuzzy_high(pet, params["c_pet"],  params["steepness"])

        # Fuzzy AND = product (differentiable; min is not)
        direct = high_car * high_pet * params["rule_strength"]

        pop = direct
        for _ in range(2):
            pop = (1.0 - params["friend_influence"]) * pop \
                + params["friend_influence"] * aggregate_friends(pop, adj, deg)
        return pop

    def loss_c(params, car, pet, adj, deg, gt):
        """MSE + small L2 regularisation on steep / friend_influence."""
        pred = model_forward_c(params, car, pet, adj, deg)
        mse  = jnp.mean((pred - gt) ** 2)
        reg  = 0.01 * (params["steepness"] ** 2 + params["friend_influence"] ** 2)
        return mse + reg


    def train_one_trial(init_params_dict: dict, n_steps: int = 600) -> dict:
        """Run gradient descent for one Optuna trial; return best accuracy + params."""

        # Convert Python scalars → JAX arrays so jax.grad can differentiate through them
        params = {k: jnp.array(float(v)) for k, v in init_params_dict.items()}

        tx    = optax.adam(learning_rate=0.01)
        state = tx.init(params)

        # FIX: argnums=0 → gradient flows into `params`, not into data arrays
        grad_fn = jax.jit(jax.value_and_grad(loss_c, argnums=0))

        best_loss   = float("inf")
        best_params = params   # FIX: initialise so it is always defined

        for _ in range(n_steps):
            loss_val, grads = grad_fn(params, pet_scores, car_scores,
                                    adj_matrix, degrees_c, popularity_gt)
            updates, state  = tx.update(grads, state, params)   # FIX: proper optax update
            params          = optax.apply_updates(params, updates)

            loss_py = float(loss_val)
            if loss_py < best_loss:
                best_loss   = loss_py
                best_params = params

        best_pred = model_forward_c(best_params, pet_scores, car_scores,
                                    adj_matrix, degrees_c)
        pred_bin  = (np.array(best_pred) > 0.5).astype(int)
        gt_bin    = (np.array(popularity_gt) > 0.5).astype(int)

        acc = float(accuracy_score(gt_bin, pred_bin))
        return {"best_loss": best_loss, "accuracy": acc, "params": best_params}

    def objective(trial: optuna.Trial) -> float:
        init_params = {
            "c_car":            trial.suggest_float("c_car",             0.30, 0.80),
            "c_pet":            trial.suggest_float("c_pet",             0.20, 0.70),
            "steepness":        trial.suggest_float("steepness",         5.0,  20.0),
            "rule_strength":    trial.suggest_float("rule_strength",     0.60,  1.00),
            "friend_influence": trial.suggest_float("friend_influence",  0.10,  0.60),
        }
        result = train_one_trial(init_params)
        return result["accuracy"]   # Optuna maximises this

    study_c = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
    study_c.optimize(objective, n_trials=40, timeout=600, show_progress_bar=False)

    print(f"\nBest accuracy : {study_c.best_value:.3f}")
    print(f"Best params   : {study_c.best_params}")

    best_result = train_one_trial(study_c.best_params, n_steps=1200)
    final_pop   = model_forward_c(best_result["params"], pet_scores, car_scores,
                                adj_matrix, degrees_c)
    trendy_c    = np.array(final_pop)

    plt.figure(figsize=(8, 6))
    nodes_c = nx.draw_networkx_nodes(G, pos, node_color=trendy_c,
                                    cmap="plasma", node_size=800)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Experiment C: Optuna-tuned Popularity Score")
    plt.colorbar(nodes_c)
    plt.tight_layout()
    plt.show()

    print("\n✅ All done.")


Download
---------

You can also download the raw notebook file for local use:
:download:`JLNN_graph_reasoning_pyreason.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_graph_reasoning_pyreason.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.
