JLNN + Knowledge Graphs: RAG-like Reasoning over FB15k-237
==============================================================

.. note::
    The interactive notebook is hosted externally to ensure the best viewing experience 
    and to allow immediate execution in the cloud.

.. grid:: 2

    .. grid-item-card::  Run in Google Colab
       :link: https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/JLNN_kg_reasoning.ipynb
       :link-type: url

       Execute the code directly in your browser without any local setup.

    .. grid-item-card::  View on GitHub
       :link: https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_kg_reasoning.ipynb
       :link-type: url

       View source code and outputs in the GitHub notebook browser.


Overview
----------

This project demonstrates a state-of-the-art **Neuro-Symbolic Reasoning** pipeline. 
It leverages **JAX Logical Neural Networks (JLNN)** to perform explainable, 
inference-based queries over the FB15k-237 Knowledge Graph dataset.

By combining semantic retrieval (RAG) with formal logical constraints, the model 
can reason about entities even when direct links are missing or noisy.

Core Methodology: The Hybrid Bridge
---------------------------------------

The system bridges the gap between connectionist (embeddings) and symbolic (rules) AI 
through a process called **Grounding**:

1. **Semantic RAG Phase:** The system uses `Sentence-Transformers` to map natural language or triple-based 
   queries to the most similar entities in the vector space.
2. **Symbolic Verification:** A "Hard Match" logic ensures that if an entity ID is explicitly present, its 
   topological neighbors in the Knowledge Graph (KG) are prioritized over semantic guesses.
3. **Predicate Mapping:** Graph relations (e.g., ``/people/person/place_of_birth``) are mapped into 
   logical predicates with truth value intervals *[L, U]*.


Theoretical Foundation: JLNN
-------------------------------

Unlike standard neural networks, JLNN outputs **Truth Value Intervals**:

* **L (Lower bound):** Evidence confirming the fact.
* **U (Upper bound):** Absence of evidence contradicting the fact.
* **Epistemic Gap (U - L):** A direct measure of the model's **uncertainty**.

Key Features
--------------

* **JAX & Flax NNX:** High-performance, differentiable logical layers.
* **Explainable AI (XAI):** Every inference can be traced back to specific KG edges or semantic similarities.
* **Robustness:** High tolerance for noise in the KG, thanks to weighted Lukasiewicz logic operators.
* **Uncertainty Quantification:** Built-in visualization of what the model knows vs. what it assumes.

Quick Start
-------------

The reasoning is performed by defining First-Order Logic rules within the *FB15kReasoner*  class:

.. code-block:: python

    # Example: A rule combining multiple predicates
    # Actor(x) ∧ ActedIn(x, y) ∧ Movie(y) → HollywoodActor(x)
    
    model = FB15kReasoner(rngs=nnx.Rngs(42))
    grounding = get_embedding_grounding("Subject_ID Relation Object_ID")
    prediction = model(grounding)


Example Code
---------------

.. code-block:: python

    '''
    try:
        import jlnn
        import jraph
        import numpyro
        from flax import nnx
        import jax.numpy as jnp
        import numpy as np
        import xarray as xr
        import pandas as pd
        import optuna
        import arviz as az
        import seaborn as sns
        import networkx as nx
        import matplotlib.pyplot as plt
        import sklearn
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
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
        !pip install sentence-transformers --quiet
        !pip install datasets --quiet
        !pip install networkx --quiet
        !pip install grain --quiet
        # Fix JAX/CUDA compatibility for 2026 in Colab
        !pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
        !pip install  scikit-learn pandas --quiet

        import os
        print("\n🔄 RESTARTING ENVIRONMENT... Please wait a second and then run the cell again.")
        os.kill(os.getpid(), 9)
        os.kill(os.getpid(), 9) # After this line, the cell stops and the environment restarts
    '''

    # 1. Imports
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import grain.python as grain
    from datasets import load_dataset
    import torch

    # JLNN importy 
    from jlnn.symbolic.compiler import LNNFormula
    from jlnn.nn.functional import weighted_and

    # Semantic retrieval
    from sentence_transformers import SentenceTransformer

    # 2. Load FB15k-237 and Build Knowledge Graph
    print("Loading FB15k-237 dataset...")
    dataset = load_dataset("KGraph/FB15k-237", trust_remote_code=True)
    train_triples = dataset['train']
    valid_triples = dataset['validation']
    test_triples = dataset['test']

    # Build directed multigraph
    kg = nx.MultiDiGraph()

    for row in train_triples:
        h, r, t = row['text'].split('\t')
        kg.add_edge(h, t, relation=r)

    print(f"Knowledge Graph built: {kg.number_of_nodes()} entities, {kg.number_of_edges()} triples")

    # 3. Embedding-based Retrieval (RAG-like) 
    print("Creating embeddings for all entities...")

    entities = list(kg.nodes())
    entity_texts = [str(e).replace("_", " ") for e in entities]
    entity_embeddings = embedder.encode(entity_texts, convert_to_tensor=True, show_progress_bar=True)

    def retrieve_relevant_entities_with_scores(query: str, k: int = 12):
        """RAG-style semantic retrieval - returns both entities and similarity scores"""
        query_emb = embedder.encode(query, convert_to_tensor=True)
        # Similarity calculation
        scores = torch.nn.functional.cosine_similarity(entity_embeddings, query_emb, dim=1)
        
        # We get the values ​​(distance/score) and indices
        top_k = scores.topk(k)
        top_k_idx = top_k.indices.cpu().numpy()
        top_k_scores = top_k.values.cpu().detach().numpy()
        
        relevant_entities = [entities[i] for i in top_k_idx]
        return relevant_entities, top_k_scores

    # 4. Dynamic Grounding into JLNN Intervals
    def clean_key(s):
        return str(s).strip('/').replace("/", "_").replace(".", "_").replace("-", "_")

    def get_embedding_grounding(query_text: str, k: int = 12):
        """
        Creates logical grounding by combining semantic search (RAG) 
        and exact search of the knowledge graph (KG) structure.
        """
        # 1. Getting relevant entities from a vector database
        relevant_entities, sim_scores = retrieve_relevant_entities_with_scores(query_text, k=k)

        # Definition of keys (predicates) that our model knows
        keys = ["Actor", "Acted_In", "Movie", "StarredIn", "Has_Genre_Drama", 
                "Good_Actor", "Born_In", "Located_In_California", "Hollywood_Actor"]
        
        # Initialize to [0.0, 0.1] - Default state "Probably No/Unknown"
        grounding = {key: jnp.array([[0.0, 0.1]]) for key in keys}

        def update_grounding(key, new_val):
            """Helper function for safe interval update (max-pooling score)"""
            val = float(new_val)
            if val < 0.4: return # We ignore semantic matches that are too weak
            
            current_l = grounding[key][0, 0]
            if val > current_l:
                # Squaring to highlight differences between certainty and noise
                boosted_val = val ** 2
                grounding[key] = jnp.array([[boosted_val, min(1.0, boosted_val + 0.1)]])

        # --- EXTRACTION OF DIRECT IDS FROM TEXT ---
        # We assume the FB15k format: "/m/subject_id /relation /m/object_id"
        query_parts = query_text.split()
        subject_id_from_query = query_parts[0] if len(query_parts) > 0 else None

        # 2. BROWSING ENTITIES AND THEIR RELATIONS IN A GRAPH
        for ent, score in zip(relevant_entities, sim_scores):
            
            # FUSE: If the found ID exactly matches the subject in the query,
            # we give it an absolute weight of 1.0, regardless of what the embedding says.
            final_score = 1.0 if ent == subject_id_from_query else score
            
            if ent in kg:
                # We traverse the actual edges (relations) in the graph for a given entity
                for _, neighbor, data in kg.edges(ent, data=True):
                    rel = data.get('relation', '').lower()
                    
                    # Mapping relations from KG to logical predicates
                    if "actor" in rel: 
                        update_grounding("Actor", final_score)
                    if "film" in rel or "movie" in rel: 
                        update_grounding("Movie", final_score)
                    if "birth" in rel: 
                        update_grounding("Born_In", final_score)
                    if "location" in rel or "california" in rel: 
                        update_grounding("Located_In_California", final_score)
                    if "star" in rel: 
                        update_grounding("StarredIn", final_score)
                    if "performance" in rel or "acted" in rel: 
                        update_grounding("Acted_In", final_score)
                    
                    # Hard-match for specific nodes (e.g. if the object is California)
                    if "california" in str(neighbor).lower():
                        update_grounding("Located_In_California", 1.0)

        # 3. TEXT TRIGGERS (searching for keywords directly in the query)
        low_query = query_text.lower()
        if "drama" in low_query: 
            update_grounding("Has_Genre_Drama", 1.0)
        if "good" in low_query or "oscar" in low_query or "award" in low_query: 
            update_grounding("Good_Actor", 0.8) # Semantic estimation of "good actor"

        return grounding

    # 5. JLNN Reasoner with Logical Rules

    class FB15kReasoner(nnx.Module):
        def __init__(self, rngs):
            # 1. Definition of logical rules
            self.rules = nnx.List([
                LNNFormula("Actor & Acted_In & Movie", rngs=rngs),
                LNNFormula("StarredIn & Has_Genre_Drama", rngs=rngs),
                LNNFormula("Actor & Born_In & Located_In_California", rngs=rngs),
            ])

            # 2. Parameter initialization (without .value deprecation)
            for rule in self.rules:
                # Setting the main gateway rule (WeightedAnd)
                if hasattr(rule.root, 'gate'):
                    gate = rule.root.gate
                    if hasattr(gate, 'weights'):
                        # Set the weights to 1.0 (ellipsis [...] instead of .value)
                        gate.weights[...] = jnp.ones_like(gate.weights)
                    
                    if hasattr(gate, 'beta'):
                        # Remove [ ] around 3.2 to make it a scalar
                        gate.beta[...] = 3.2
                
                # Setting individual predicates within a rule
                for pred in rule.predicates.values():
                    if hasattr(pred, 'predicate') and hasattr(pred.predicate, 'linear'):
                        lin = pred.predicate.linear
                        lin.kernel[...] = jnp.ones_like(lin.kernel)
                        # Here jnp.full_like will preserve the original shape, which is correct
                        lin.bias[...] = jnp.full_like(lin.bias, -0.6)

        def __call__(self, grounding):
            """Inference: Will push grounding through all rules"""
            results = [rule(grounding) for rule in self.rules]
            final_results = []
            
            for i, r in enumerate(results):
                # Dimension handling (ensuring the form [1, 2] for intervals)
                r_cleaned = r[:, -1, :] if r.ndim == 3 else r.reshape(1, 2)
                # Debug log that we saw in the console
                # print(f"Rule {i} value: {r_cleaned}") 
                final_results.append(r_cleaned)
                
            # Final aggregation of all rules into one result
            ready_to_go = jnp.stack(final_results, axis=-2)
            
            # Last weighted_and with high beta for total output
            return weighted_and(
                ready_to_go, 
                weights=jnp.array([1.0, 1.0, 1.0]), 
                beta=3.2
            )

    rngs = nnx.Rngs(42)
    model = FB15kReasoner(rngs)

    # 6. Grain Data Loader
    class KGSampler(grain.RandomAccessDataSource):
        def __init__(self, triples): self.triples = triples
        def __len__(self): return len(self.triples)
        def __getitem__(self, idx): return self.triples[idx]

    data_source = KGSampler(train_triples)
    sampler = grain.IndexSampler(num_records=len(data_source), shard_options=grain.NoSharding(), shuffle=True, seed=42)
    loader = grain.DataLoader(data_source=data_source, sampler=sampler, worker_count=1)

    # 7. Downloaded Data Content List and Retrieval
    def inspect_data_and_retrieval(batch_item, k=5):
        """
        It prints the raw input and shows which entities and relations to it were traced.
        """
        text_query = batch_item['text']
        parts = text_query.split('\t')
        
        print("📄 RAW INPUT (FB15k Triple):")
        print(f"   Subject:   {parts[0]}")
        print(f"   Relation:  {parts[1]}")
        print(f"   Object:    {parts[2]}")
        print("-" * 50)
        
        # Semantic entity search
        relevant_nodes = retrieve_relevant_entities_with_scores(text_query, k=k)
        
        print(f"🔍 RAG RETRIEVAL (Top-{k} found entities in the graph):")
        for i, node in enumerate(relevant_nodes):
            # Clean up the name for the listing
            clean_name = str(node).replace("/m/", "entity ").replace("_", " ")
            print(f"   {i+1}. {node} ({clean_name})")
            
            # Example of the relationships that this entity has in the graph
            if node in kg:
                edges = list(kg.edges(node, data=True))[:2] # only the first 2 samples
                for _, target, data in edges:
                    print(f"      └─ Relation: {data['relation']} ➔ {target}")
        
        print("-" * 50)

    print("🚀 Data analysis in the pipeline:")
    for i, batch in enumerate(loader):
        if i >= 3: break
        print(f"\nSAMPLE #{i+1}:")
        inspect_data_and_retrieval(batch)

    # 8. Inference with Embedding Retrieval
    print("\n🚀 Running JLNN Inference with Embedding Retrieval")
    print("="*85)

    # 1. Here we create an empty list for the results
    results_list = []

    for i, batch in enumerate(loader):
        if i >= 6: 
            break
        text = batch['text']
        grounding = get_embedding_grounding(text, k=12)
        print(f"DEBUG Grounding for Sample {i}: {grounding}") # HERE YOU WILL SEE THE TRUTH
        prediction = model(grounding)

        # 2. HERE WE SAVE THE RESULT IN A LIST
        results_list.append(prediction)
        
        print(f"SAMPLE #{i+1}: {text}")
        print(f"JLNN Output [L, U] → {prediction}")
        print("-" * 85)

    # 9. Visualization
    def plot_epistemic_gap(grounding):
        labels = list(grounding.keys())
        lowers = [float(grounding[k][0,0]) for k in labels]
        uppers = [float(grounding[k][0,1]) for k in labels]
        gaps = np.array(uppers) - np.array(lowers)
        
        plt.figure(figsize=(14, 8))
        plt.barh(labels, gaps, left=lowers, color='skyblue', edgecolor='black')
        plt.xlabel("Truth Value Interval [L, U]")
        plt.title("JLNN Knowledge Graph Grounding – Epistemic Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    plot_epistemic_gap(grounding)

    def plot_jlnn_results(all_results):
        """
        Plots the final intervals [L, U] for all samples.
        
        Args:
            all_results (list): A list of jnp.arrays in the form [[L, U]] for each sample.
        """
        labels = [f"Sample #{i+1}" for i in range(len(all_results))]
        lowers = [float(r[0, 0]) for r in all_results]
        uppers = [float(r[0, 1]) for r in all_results]
        
        # Interval width (epistemic uncertainty)
        widths = np.array(uppers) - np.array(lowers)
        
        plt.figure(figsize=(12, 6))
        
        # Draw horizontal bars (intervals)
        bars = plt.barh(labels, widths, left=lowers, color='lightgreen', edgecolor='black', alpha=0.7)
        
        # Adding a vertical line for the average value (center of the interval) for better orientation
        for i, (l, u) in enumerate(zip(lowers, uppers)):
            mid = (l + u) / 2
            plt.plot([mid, mid], [i - 0.4, i + 0.4], color='darkgreen', lw=2)

        plt.xlim(0, 1.05) # Truth range 0 to 1
        plt.xlabel("Truth Value Interval [L, U]")
        plt.title("JLNN Final Reasoning Results - Comparison Across Samples")
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        # "Noise" enhancement (Sample #5)
        plt.gca().get_yticklabels()[4].set_color('red') # Sample 5 is at index 4
        plt.gca().get_yticklabels()[4].set_weight('bold')

        plt.tight_layout()
        plt.show()

    plot_jlnn_results(results_list)


Interpretation of Visual Results
-----------------------------------

The project includes an epistemic visualization suite:

* **Right-shifted Bars:** Facts verified by the Knowledge Graph.
* **Left-shifted Bars:** Irrelevant or refuted information (e.g., noise filtering).
* **Wide Intervals:** Areas where the KG is incomplete, prompting for further data collection.

Future Work
--------------

* **Supervised Rule Learning:** Training rule weights and gate biases using JAX optimizers.
* **Recursive Reasoning:** Implementing multi-hop backward chaining for complex queries.

Download
-----------

You can also download the raw notebook file for local use:
:download:`JLNN_kg_reasoning.ipynb <https://github.com/RadimKozl/JLNN/blob/main/examples/JLNN_kg_reasoning.ipynb>`

.. tip::
   To run the notebook locally, make sure you have installed the package using ``pip install -e .[test]``.