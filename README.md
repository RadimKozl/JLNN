# JLNN – JAX Logical Neural Networks
-------------------------------------

<p align="center">
  <img src="docs/source/_static/jlnn_diagram.png" width="400" alt="JLNN Logo">
</p>

Neuro-symbolic framework for interval-based Łukasiewicz logic built on **JAX** + **Flax NNX**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RadimKozl/JLNN/blob/main/examples/Jax_lnn_base.ipynb)

JLNN enables turning symbolic logical rules into differentiable neural networks for training on data while maintaining interpretability and logical consistency.

## **Features**

- **Interval truth values [L, U]**: Full support for uncertainty and contradiction modeling.
- **Weighted logical gates**: AND, OR, XOR, Implication, and NOT with Łukasiewicz semantics.
- **Symbolic Compiler**: Compile formulas like `"0.8::A & B -> C"` directly to NNX graphs.
- **Temporal Logic**: Experimental support for temporal operators (G, F, X).
- **Logical Constraints**: Built-in enforcement of axioms (e.g., weights $w \geq 1.0$).
- **High Performance**: JIT-compilation and hardware acceleration via JAX.
- **Interoperability**: Export trained models to ONNX, StableHLO, or PyTorch.

## **Installation**

```bash
# From PyPI
pip install jax-lnn

# From GitHub
pip install git+[https://github.com/RadimKozl/JLNN.git](https://github.com/RadimKozl/JLNN.git)

# For development
git clone [https://github.com/RadimKozl/JLNN.git](https://github.com/RadimKozl/JLNN.git)
cd JLNN
uv sync  # or pip install -e ".[test]"
```

## **Quickstart**

```python

import jax
import jax.numpy as jnp
from flax import nnx
from jlnn.symbolic.compiler import LNNFormula
from jlnn.nn.constraints import apply_constraints
from jlnn.training.losses import total_lnn_loss, logical_mse_loss, contradiction_loss
from jlnn.storage.checkpoints import save_checkpoint, load_checkpoint
import optax

# 1. Define and compile the formula
model = LNNFormula("0.8::A & B -> C", nnx.Rngs(42))

# 2. Ground inputs (including initial state for C)
inputs = {
    "A": jnp.array([[0.9]]),
    "B": jnp.array([[0.7]]),
    "C": jnp.array([[0.5]])   # MANDATORY – consequent must have grounding!
}

target = jnp.array([[0.6, 0.85]])

# 3. Loss function
def loss_fn(model, inputs, target):
    pred = model(inputs)
    pred = jnp.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)  # protection against NaN
    return total_lnn_loss(pred, target)

# 4. Initialize Optimizer
optimizer = nnx.Optimizer(
    model,
    wrt=nnx.Param,
    tx=optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=0.001)
    )
)

# 5. Training Step
@nnx.jit
def train_step(model, optimizer, inputs, target):
    # Gradients to the model – closure is traceable (inputs/target are arrays)
    grads = nnx.grad(lambda m: loss_fn(m, inputs, target))(model)

    # Loss before update (for debug)
    loss = loss_fn(model, inputs, target)

    optimizer.update(model, grads)
    apply_constraints(model)

    final_loss = loss_fn(model, inputs, target)
    final_pred = model(inputs)

    return loss, final_loss, final_pred

print("=== Starting training ===")
steps = 50
for step in range(steps):
    loss, final_loss, pred = train_step(model, optimizer, inputs, target)

    print(f"Step {step:3d} | Loss before/after constraints: {loss:.6f} → {final_loss:.6f}")
    print(f"Prediction: {pred}")
    print("─" * 60)

    if jnp.isnan(final_loss).any():
        print("❌ NaN detected! Stopping.")
        break

print("=== Training completed ===")

# 6. Result after training

final_pred = model(inputs)
print("\nFinal prediction after training:")
print(final_pred)

print(f"\nTarget interval: {target}")
print(f"Final loss: {total_lnn_loss(final_pred, target):.6f}")

```
See the introductory Jupyter notebook: [Jax_lnn_base.ipynb](https://github.com/RadimKozl/JLNN/blob/main/examples/Jax_lnn_base.ipynb)

## **Acknowledgments & Inspiration**

JLNN is inspired by and builds upon the foundations laid by several pioneering neuro-symbolic projects:

- [***LNN***](https://github.com/IBM/LNN) (IBM Research) – The primary inspiration for interval-based logical neural networks.
- [***LTNtorch***](https://github.com/tommasocarraro/LTNtorch) – Logic Tensor Networks implementation in PyTorch.
- [***PyReason***](https://github.com/lab-v2/pyreason) – Software for open-world temporal logic reasoning.

## **Documentation**

- [***Online Documentation***](https://radimkozl.github.io/JLNN/)
- [***Introductory Tutorial***](https://radimkozl.github.io/JLNN/tutorials/introduction_tutorial.html)
- [***API Reference***](https://radimkozl.github.io/JLNN/)

## **License**

This project is licensed under the MIT License - see the [***LICENSE***](https://radimkozl.github.io/JLNN/license.html) file for details.

