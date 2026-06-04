"""
Example: JAX training loop on synthetic MNIST-style data.

Demonstrates the JAX functional training pattern:
  - Explicit PRNG key management (no global random state)
  - jax.jit compilation for speed
  - Immutable TrainState updates
  - Functional gradient computation with jax.value_and_grad

The data is synthetic (random arrays) so this runs without downloading
the real MNIST dataset. Replace with a real dataloader for actual use.

Run: python examples/jax_mnist.py
"""

import sys

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:
    print("JAX/optax not installed. pip install jax jaxlib optax")
    print("(Showing code structure without execution)")
    # Still print the code logic for educational purposes
    print("""
Conceptual JAX training loop:

1. Define model as a pure function: forward(params, x) -> logits
2. Initialize: params = model.init(rng_key, dummy_input)
3. Compile train step: train_step = jax.jit(lambda state, batch: ...)
4. Loop: for batch in data: state, metrics = train_step(state, batch)

Key difference vs PyTorch:
  - No model.train() / model.eval(), no mutable state
  - No optimizer.zero_grad(), gradients are explicit outputs
  - No loss.backward(), jax.value_and_grad handles this
  - Every function is pure: same inputs → same outputs, always
""")
    sys.exit(0)

import numpy as np


# ── Simple MLP defined as pure functions ──────────────────────────────────────
# In JAX, a "model" is just:
#   init_fn(key, x) → params
#   apply_fn(params, x) → output
#
# We define this manually here instead of using Flax to show the raw pattern.

def init_mlp_params(key, input_dim: int, hidden_dim: int, num_classes: int):
    """
    Initialize MLP parameters.

    JAX doesn't have a global random state, you pass an explicit PRNG key.
    jax.random.split creates two new keys from one, enabling independent
    random streams without side effects.
    """
    k1, k2 = jax.random.split(key)

    # He initialization: scale by sqrt(2 / fan_in)
    w1 = jax.random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros(hidden_dim)

    w2 = jax.random.normal(k2, (hidden_dim, num_classes)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros(num_classes)

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def mlp_forward(params, x):
    """
    Forward pass: pure function, no side effects.

    Takes params dict and input x, returns logits.
    This is exactly what jax.jit and jax.grad can differentiate.
    """
    # Layer 1 + ReLU
    h = jnp.dot(x, params["w1"]) + params["b1"]
    h = jax.nn.relu(h)

    # Layer 2 (output logits)
    logits = jnp.dot(h, params["w2"]) + params["b2"]
    return logits


def cross_entropy_loss(params, x, labels):
    """
    Cross-entropy loss.

    This function is fully pure, no global state, no mutation.
    jax.value_and_grad(cross_entropy_loss)(params, x, labels)
    returns (loss_value, grad_wrt_params).
    """
    logits = mlp_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# ── JIT-compiled train step ────────────────────────────────────────────────────

@jax.jit
def train_step_jit(params, opt_state, x, y, tx):
    """
    Single gradient update step.

    @jax.jit: Traces this function once, compiles to XLA.
    Subsequent calls are ~10-100x faster than Python eager execution.

    The pattern:
      1. Compute loss AND gradients in one pass (jax.value_and_grad)
      2. Ask optimizer for parameter updates
      3. Apply updates to get new params
      4. Return new params + new opt_state (immutable update)
    """
    # jax.value_and_grad differentiates w.r.t. the first argument (params)
    loss_val, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)

    # optax optimizer: pure functions returning (updates, new_opt_state)
    updates, new_opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_val


@jax.jit
def eval_step_jit(params, x, y):
    """Evaluation: forward pass + accuracy, no gradient computation."""
    logits = mlp_forward(params, x)
    predicted = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predicted == y)
    loss = cross_entropy_loss(params, x, y)
    return {"loss": loss, "accuracy": accuracy}


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_synthetic_dataset(num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10, seed: int = 0):
    """
    Generate synthetic MNIST-style data.

    Replace with a real dataloader for actual training:
      from datasets import load_dataset
      mnist = load_dataset("mnist")
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, input_dim).astype(np.float32)
    y = rng.randint(0, num_classes, num_samples).astype(np.int32)
    return jnp.array(x), jnp.array(y)


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    print("JAX MNIST-style Training Demo")
    print("=" * 45)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # Config
    INPUT_DIM = 784    # 28x28 pixels, flattened
    HIDDEN_DIM = 256
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3
    NUM_TRAIN = 5000
    NUM_VAL = 1000

    # Data
    x_train, y_train = make_synthetic_dataset(NUM_TRAIN, INPUT_DIM, NUM_CLASSES, seed=42)
    x_val, y_val = make_synthetic_dataset(NUM_VAL, INPUT_DIM, NUM_CLASSES, seed=99)
    print(f"Train: {x_train.shape}  Val: {x_val.shape}")

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = init_mlp_params(key, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
    print(f"Params initialized: {sum(v.size for v in jax.tree_util.tree_leaves(params)):,} parameters")

    # Initialize optimizer
    # optax.adam is a pure function that returns (init_fn, update_fn)
    tx = optax.adam(LEARNING_RATE)
    opt_state = tx.init(params)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}...")
    num_batches = NUM_TRAIN // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []

        # Shuffle training data each epoch
        perm_key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(perm_key, NUM_TRAIN)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for i in range(num_batches):
            x_batch = x_shuffled[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_batch = y_shuffled[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            # train_step returns new (immutable) state, old params/opt_state unchanged
            params, opt_state, loss = train_step_jit(params, opt_state, x_batch, y_batch, tx)
            epoch_losses.append(float(loss))

        # Evaluate on validation set
        val_metrics = eval_step_jit(params, x_val, y_val)

        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        print(
            f"  Epoch {epoch + 1}/{NUM_EPOCHS}  "
            f"train_loss={mean_train_loss:.4f}  "
            f"val_loss={float(val_metrics['loss']):.4f}  "
            f"val_acc={float(val_metrics['accuracy']):.4f}"
        )

    print("\nTraining complete.")
    print("Note: accuracy on random data stays near 1/num_classes ≈ 0.10")
    print("With real MNIST data and more epochs, you'd expect >97% accuracy.")


if __name__ == "__main__":
    main()
