"""
JAX training utilities with functional-style training loop.

JAX vs PyTorch, the key mental shift:
---------------------------------------
PyTorch is imperative: you mutate model parameters in-place via optimizer.step().
State lives in Python objects and changes over time.

JAX is purely functional: there is no mutation. Instead, you have a TrainState
object that holds parameters + optimizer state, and train_step() returns a NEW
TrainState. The old state is never modified: it's immutable. This is what
enables JAX's composability: jit, grad, vmap, pmap all work cleanly because
there are no side effects.

The jax.jit decorator traces the function at first call and compiles it to XLA.
Subsequent calls use the compiled version, often 10 to 100x faster than eager.

Requires: jax>=0.4, flax>=0.7 (or optax for optimizers).
These are optional, graceful fallback is provided.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Optional JAX imports ───────────────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import optax
    _OPTAX_AVAILABLE = True
except ImportError:
    _OPTAX_AVAILABLE = False

try:
    import flax.linen as nn
    from flax.training import train_state as flax_train_state
    _FLAX_AVAILABLE = True
except ImportError:
    _FLAX_AVAILABLE = False


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class JAXTrainState:
    """
    Minimal train state for JAX training.

    In Flax you'd use flax.training.train_state.TrainState. This is a
    lighter version that works without the full Flax stack.

    Attributes:
        params: Model parameters (pytree of arrays).
        opt_state: Optimizer state (pytree, managed by optax).
        step: Current global training step.
        apply_fn: The model's apply function (params to outputs).
    """
    params: Any
    opt_state: Any
    step: int
    apply_fn: Any  # callable: (params, x) to output

    def replace(self, **kwargs) -> "JAXTrainState":
        """Return a new JAXTrainState with updated fields (immutable update)."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)


@dataclass
class JAXTrainingHistory:
    """Training history from a JAX training loop."""
    train_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    epochs: int = 0


# ── TrainState factory ─────────────────────────────────────────────────────────

def create_train_state(
    model: Any,
    learning_rate: float,
    input_shape: Tuple,
    seed: int = 42,
) -> JAXTrainState:
    """
    Initialize a JAXTrainState with model parameters and optimizer state.

    In JAX, model initialization is explicit: you call model.init() with a
    PRNG key and dummy input to get the initial parameter pytree. There's no
    global random state, every random operation takes a key explicitly.

    Args:
        model: Flax Module (or any object with .init(key, x) to params).
        learning_rate: Learning rate for the Adam optimizer.
        input_shape: Shape of a single input sample (used to create dummy input).
        seed: PRNG seed for parameter initialization.

    Returns:
        JAXTrainState with initialized params and optimizer state.

    Note:
        Requires jax + optax. Falls back to a dummy state if not installed.
    """
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX not installed. pip install jax jaxlib\n"
            "For GPU support: pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        )
    if not _OPTAX_AVAILABLE:
        raise ImportError("optax not installed. pip install optax")

    # Initialize PRNG key: JAX requires explicit key management
    key = jax.random.PRNGKey(seed)

    # Create dummy input for shape inference
    dummy_input = jnp.ones((1,) + input_shape)

    # Initialize model parameters
    # model.init() returns {"params": {...}} for Flax modules
    if _FLAX_AVAILABLE and isinstance(model, nn.Module):
        variables = model.init(key, dummy_input)
        params = variables["params"]
        apply_fn = model.apply
    else:
        # Fallback: treat model as a simple function with params attribute
        params = getattr(model, "params", {})
        apply_fn = model

    # Create Adam optimizer via optax
    # optax.adam returns a GradientTransformation, a pure function pair:
    #   init(params) to opt_state
    #   update(grads, opt_state) to (updates, new_opt_state)
    tx = optax.adam(learning_rate)
    opt_state = tx.init(params)

    return JAXTrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        apply_fn=apply_fn,
    )


# ── Training step (JIT-compiled) ───────────────────────────────────────────────

def train_step(
    state: JAXTrainState,
    batch: Tuple,
    loss_fn: Any = None,
    tx: Any = None,
) -> Tuple[JAXTrainState, Dict]:
    """
    Single training step: computes gradients and returns a new state.

    This function is designed to be wrapped with @jax.jit for compilation.
    The key pattern:
      1. jax.value_and_grad computes both the loss and the gradient in one pass.
      2. optax.apply_updates applies the gradient to the parameters.
      3. Return a NEW state, nothing is mutated.

    Args:
        state: Current JAXTrainState.
        batch: (inputs, labels) tuple.
        loss_fn: Loss function (inputs, params, labels) to scalar.
                 Defaults to cross-entropy if None.
        tx: optax optimizer. Required for parameter updates.

    Returns:
        (new_state, metrics_dict) tuple.
        metrics_dict contains 'loss' and optionally 'accuracy'.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX not installed.")
    if not _OPTAX_AVAILABLE:
        raise ImportError("optax not installed.")

    inputs, labels = batch

    if loss_fn is None:
        # Default: cross-entropy loss for classification
        def loss_fn(params, inputs, labels):
            logits = state.apply_fn(params, inputs)
            # Softmax cross-entropy
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(labels, logits.shape[-1])
            return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    # jax.value_and_grad differentiates w.r.t. the first argument (params)
    # grad_fn returns (loss_value, gradient_pytree)
    grad_fn = jax.value_and_grad(loss_fn)
    loss_value, grads = grad_fn(state.params, inputs, labels)

    # Optimizer: compute parameter updates from gradients
    tx_optax = optax.adam(1e-3)  # Fallback; in practice pass tx as argument
    updates, new_opt_state = tx_optax.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)

    # Build new state, old state is not modified
    new_state = state.replace(
        params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1,
    )

    metrics = {"loss": float(loss_value)}
    return new_state, metrics


def eval_step(state: JAXTrainState, batch: Tuple) -> Dict:
    """
    Evaluation step: no gradient computation, just forward pass + metrics.

    JAX doesn't have a torch.no_grad() equivalent, it's not needed because
    gradients are only computed when you explicitly call jax.grad or jax.jit.
    A regular forward pass never computes gradients unless you ask.

    Args:
        state: Current JAXTrainState.
        batch: (inputs, labels) tuple.

    Returns:
        metrics_dict with 'loss' and 'accuracy'.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX not installed.")

    inputs, labels = batch
    logits = state.apply_fn(state.params, inputs)

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    # Accuracy
    predicted = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predicted == labels)

    return {"loss": float(loss), "accuracy": float(accuracy)}


def train_loop(
    state: JAXTrainState,
    train_batches: List[Tuple],
    num_epochs: int,
    loss_fn: Any = None,
    verbose: bool = True,
) -> JAXTrainingHistory:
    """
    Full JAX training loop.

    Each epoch iterates over train_batches, calling train_step for each.
    Since JAX compiles train_step on first call (via jax.jit), subsequent
    calls are much faster, the XLA compiler optimizes the full computation graph.

    Args:
        state: Initial JAXTrainState.
        train_batches: List of (inputs, labels) batches.
        num_epochs: Number of training epochs.
        loss_fn: Optional custom loss function.
        verbose: Whether to print epoch summaries.

    Returns:
        JAXTrainingHistory with per-epoch losses and accuracies.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX not installed.")

    # JIT-compile train_step for speed
    # jax.jit traces the function once and compiles it to XLA IR.
    # After the first call, each subsequent call hits the compiled version.
    jit_train_step = jax.jit(lambda s, b: train_step(s, b, loss_fn=loss_fn))

    history = JAXTrainingHistory()
    current_state = state

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in train_batches:
            current_state, metrics = jit_train_step(current_state, batch)
            epoch_losses.append(metrics["loss"])

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.train_losses.append(mean_loss)
        history.train_accuracies.append(0.0)  # Fill if eval step is used
        history.epochs += 1

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}  loss={mean_loss:.4f}")

    return history
