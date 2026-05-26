"""
Distributed training configuration and validation.

Covers the four main distributed strategies:
  - DataParallel (DP): Simple single-node multi-GPU. All GPUs share the same
    model but get different data shards. Gradient sync after each step.
    Simple but not the most efficient due to Python GIL contention.

  - DistributedDataParallel (DDP): Multi-process, each process owns one GPU.
    Much better performance than DP. The standard choice for multi-GPU training
    on a single or multi-node setup.

  - FSDP (Fully Sharded Data Parallel): Shards model parameters, gradients,
    and optimizer state across GPUs. Enables training models too large to fit
    on a single GPU. Used by Meta's LLaMA training.

  - ModelParallel: Splits the model across GPUs (e.g., first half on GPU 0,
    second half on GPU 1). Used for large models but requires manual pipeline
    design. Usually combined with DDP in practice (tensor parallelism).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DistributedStrategy(str, Enum):
    DATA_PARALLEL = "DataParallel"
    DDP = "DDP"
    FSDP = "FSDP"
    MODEL_PARALLEL = "ModelParallel"


class Backend(str, Enum):
    NCCL = "nccl"    # GPU-optimized collective operations (recommended for CUDA)
    GLOO = "gloo"    # CPU fallback, also used for CPU+GPU mixed setups
    MPI = "mpi"      # High-performance inter-node communication


class MixedPrecision(str, Enum):
    FP32 = "fp32"    # Full precision — most memory, most stable
    FP16 = "fp16"    # Half precision — 2x memory savings, may need loss scaling
    BF16 = "bf16"    # Brain float — same dynamic range as fp32, no loss scaling needed


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.

    Validates that the combination of strategy, backend, precision, and
    parallelism settings is internally consistent.

    Attributes:
        strategy: Distributed training strategy.
        num_gpus: Number of GPUs per node.
        num_nodes: Number of training nodes.
        backend: Communication backend.
        mixed_precision: Numeric precision for weights and activations.
        gradient_accumulation_steps: Number of steps before optimizer update.
                                      Effective batch size = batch_size × grad_accum.
        gradient_checkpointing: Whether to trade compute for memory by recomputing
                                 activations during backward pass.
        find_unused_parameters: DDP flag for models with conditional forward paths.
    """
    strategy: DistributedStrategy = DistributedStrategy.DDP
    num_gpus: int = 1
    num_nodes: int = 1
    backend: Backend = Backend.NCCL
    mixed_precision: MixedPrecision = MixedPrecision.FP32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    find_unused_parameters: bool = False

    @property
    def world_size(self) -> int:
        """Total number of processes = num_gpus × num_nodes."""
        return self.num_gpus * self.num_nodes

    @property
    def effective_batch_multiplier(self) -> int:
        """Effective batch size multiplier vs single-GPU."""
        return self.world_size * self.gradient_accumulation_steps

    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of error strings.

        Returns:
            Empty list if valid. List of error messages if invalid.
        """
        errors = []

        if self.num_gpus < 1:
            errors.append("num_gpus must be >= 1")
        if self.num_nodes < 1:
            errors.append("num_nodes must be >= 1")
        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")

        # FSDP requires DDP-style multi-process — needs nccl backend
        if self.strategy == DistributedStrategy.FSDP and self.backend == Backend.GLOO:
            errors.append("FSDP requires nccl backend for GPU communication")

        # DataParallel doesn't work across nodes (single-process, multi-GPU only)
        if self.strategy == DistributedStrategy.DATA_PARALLEL and self.num_nodes > 1:
            errors.append(
                "DataParallel is single-node only. Use DDP or FSDP for multi-node training."
            )

        # BF16 requires Ampere or newer (compute capability >= 8.0)
        # We can't check GPU arch here, so just warn in the comment.
        # FP16 on CPU is not useful
        if self.mixed_precision == MixedPrecision.FP16 and self.num_gpus == 0:
            errors.append("FP16 is not meaningful on CPU-only training")

        # FSDP + MODEL_PARALLEL is an unusual combination
        if (self.strategy == DistributedStrategy.FSDP and
                self.strategy == DistributedStrategy.MODEL_PARALLEL):
            errors.append("Cannot use FSDP and ModelParallel simultaneously")

        return errors

    def is_valid(self) -> bool:
        """Return True if configuration has no validation errors."""
        return len(self.validate()) == 0

    def to_torch_config(self) -> Dict:
        """
        Convert to a dict suitable for initializing PyTorch distributed training.

        Returns:
            Dict with keys: backend, init_method, world_size, rank placeholder,
            find_unused_parameters, mixed_precision settings.
        """
        config: Dict = {
            "backend": self.backend.value,
            "init_method": "env://",  # Standard: read MASTER_ADDR, MASTER_PORT from env
            "world_size": self.world_size,
            "rank": None,  # Set at launch time by torchrun / SLURM
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

        # Mixed precision
        if self.mixed_precision == MixedPrecision.FP16:
            config["amp_dtype"] = "float16"
            config["use_amp"] = True
        elif self.mixed_precision == MixedPrecision.BF16:
            config["amp_dtype"] = "bfloat16"
            config["use_amp"] = True
        else:
            config["amp_dtype"] = "float32"
            config["use_amp"] = False

        # FSDP-specific config
        if self.strategy == DistributedStrategy.FSDP:
            config["fsdp_config"] = {
                "sharding_strategy": "FULL_SHARD",  # shard params + grads + optimizer state
                "cpu_offload": False,  # offload to CPU RAM to fit even larger models
                "backward_prefetch": "BACKWARD_PRE",
                "mixed_precision_policy": config["amp_dtype"],
            }

        return config

    def to_jax_config(self) -> Dict:
        """
        Convert to a dict suitable for JAX/pjit distributed setup.

        JAX uses a device mesh model (pjit/shard_map) rather than process groups.
        This config provides the mesh axes and sharding annotations.

        Returns:
            Dict with JAX-compatible distributed configuration.
        """
        return {
            "num_devices": self.world_size,
            "mesh_axes": ("data",) if self.strategy != DistributedStrategy.MODEL_PARALLEL else ("data", "model"),
            "data_axis": "data",
            "model_axis": "model" if self.strategy == DistributedStrategy.MODEL_PARALLEL else None,
            "dtype": {
                MixedPrecision.FP32: "float32",
                MixedPrecision.FP16: "float16",
                MixedPrecision.BF16: "bfloat16",
            }[self.mixed_precision],
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

    def estimate_memory_gb(
        self,
        model_params_millions: float,
        batch_size: int,
        sequence_length: int = 512,
    ) -> float:
        """
        Rough estimate of GPU memory requirement per device.

        Formula (simplified):
          params_bytes = params_M * 1e6 * bytes_per_param
          gradients = params_bytes (same size as params)
          optimizer_state = 2x params_bytes for Adam (momentum + variance)
          activations = batch_size * seq_len * hidden_dim * 4 bytes (rough)

        This is a rough estimate — actual usage depends on model architecture,
        activation checkpointing, and framework overhead.

        Args:
            model_params_millions: Model parameter count in millions (e.g., 7000 for 7B).
            batch_size: Per-device batch size.
            sequence_length: Sequence length (relevant for transformers).

        Returns:
            Estimated GB per device.
        """
        bytes_per_param = {
            MixedPrecision.FP32: 4,
            MixedPrecision.FP16: 2,
            MixedPrecision.BF16: 2,
        }[self.mixed_precision]

        params_bytes = model_params_millions * 1e6 * bytes_per_param

        if self.strategy in (DistributedStrategy.FSDP,):
            # FSDP shards params + grads + optimizer state across world_size
            shard_factor = self.world_size
        elif self.strategy == DistributedStrategy.MODEL_PARALLEL:
            shard_factor = max(self.num_gpus, 1)
        else:
            # DDP and DP replicate the model on every GPU
            shard_factor = 1

        # Params + grads per device (sharded if FSDP)
        model_bytes_per_device = params_bytes / shard_factor

        # Adam optimizer state: 2x params (fp32 master weights + momentum + variance)
        optimizer_bytes = params_bytes * 3 * 4 / shard_factor  # Always fp32 for Adam states

        # Rough activation estimate (3 bytes per token per layer is a common heuristic)
        # Simplified: batch * seq_len * 2048 (hidden_dim estimate) * 4 bytes
        activation_bytes = batch_size * sequence_length * 2048 * 4

        if self.gradient_checkpointing:
            # Gradient checkpointing recomputes activations → only O(sqrt(layers)) stored
            activation_bytes *= 0.1  # rough approximation

        total_bytes = model_bytes_per_device + optimizer_bytes + activation_bytes
        return total_bytes / (1024 ** 3)  # Convert to GB

    def __str__(self) -> str:
        return (
            f"DistributedConfig(\n"
            f"  strategy={self.strategy.value}  num_gpus={self.num_gpus}  "
            f"num_nodes={self.num_nodes}  world_size={self.world_size}\n"
            f"  backend={self.backend.value}  precision={self.mixed_precision.value}\n"
            f"  grad_accum={self.gradient_accumulation_steps}  "
            f"effective_batch_mult={self.effective_batch_multiplier}\n"
            f")"
        )
