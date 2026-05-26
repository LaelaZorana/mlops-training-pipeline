"""
Example: Configuring distributed training for different scenarios.

Shows how to use DistributedConfig for:
  1. Single GPU baseline
  2. Multi-GPU DDP on one node
  3. FSDP for large models (7B+ parameters)
  4. Multi-node DDP
  5. Mixed precision with gradient accumulation

Run: python examples/distributed_setup.py
"""

from pipeline.distributed_config import (
    DistributedConfig,
    DistributedStrategy,
    Backend,
    MixedPrecision,
)


def show_config(name: str, config: DistributedConfig):
    """Print config summary, validation status, and memory estimate."""
    print(f"\n{'─' * 55}")
    print(f"Scenario: {name}")
    print(f"{'─' * 55}")
    print(config)

    errors = config.validate()
    if errors:
        print(f"Validation: FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"Validation: OK")

    # Memory estimate for a 7B parameter model
    mem_7b = config.estimate_memory_gb(
        model_params_millions=7_000,
        batch_size=4,
        sequence_length=2048,
    )
    print(f"Estimated GPU memory (7B params, batch=4, seq=2048): {mem_7b:.1f} GB/device")

    torch_cfg = config.to_torch_config()
    print(f"torch config keys: {list(torch_cfg.keys())}")


def main():
    print("Distributed Training Configuration Examples")
    print("=" * 55)

    # ── Scenario 1: Single GPU ─────────────────────────────────────────────────
    show_config(
        "Single GPU, fp32",
        DistributedConfig(
            strategy=DistributedStrategy.DDP,
            num_gpus=1,
            num_nodes=1,
            backend=Backend.NCCL,
            mixed_precision=MixedPrecision.FP32,
        ),
    )

    # ── Scenario 2: 4-GPU DDP, bf16 ───────────────────────────────────────────
    show_config(
        "4× GPU DDP with bf16 (recommended for Ampere+)",
        DistributedConfig(
            strategy=DistributedStrategy.DDP,
            num_gpus=4,
            num_nodes=1,
            backend=Backend.NCCL,
            mixed_precision=MixedPrecision.BF16,
            gradient_accumulation_steps=4,  # effective batch = 4×GPUs×accum = 16x
        ),
    )

    # ── Scenario 3: FSDP for large model ──────────────────────────────────────
    show_config(
        "8× GPU FSDP for 7B model (full sharding)",
        DistributedConfig(
            strategy=DistributedStrategy.FSDP,
            num_gpus=8,
            num_nodes=1,
            backend=Backend.NCCL,
            mixed_precision=MixedPrecision.BF16,
            gradient_checkpointing=True,  # Trade compute for memory
            gradient_accumulation_steps=2,
        ),
    )

    # ── Scenario 4: Multi-node DDP ─────────────────────────────────────────────
    show_config(
        "Multi-node: 4 nodes × 8 GPUs = 32 GPUs total",
        DistributedConfig(
            strategy=DistributedStrategy.DDP,
            num_gpus=8,
            num_nodes=4,
            backend=Backend.NCCL,
            mixed_precision=MixedPrecision.BF16,
        ),
    )

    # ── Scenario 5: Invalid config (DataParallel multi-node) ──────────────────
    show_config(
        "INVALID: DataParallel across nodes",
        DistributedConfig(
            strategy=DistributedStrategy.DATA_PARALLEL,
            num_gpus=4,
            num_nodes=2,  # This is not valid for DataParallel
            backend=Backend.GLOO,
            mixed_precision=MixedPrecision.FP32,
        ),
    )

    # ── JAX config ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print("JAX config translation:")
    jax_config = DistributedConfig(
        strategy=DistributedStrategy.DDP,
        num_gpus=4,
        mixed_precision=MixedPrecision.BF16,
    ).to_jax_config()
    for k, v in jax_config.items():
        print(f"  {k}: {v}")

    # ── Memory scaling demonstration ──────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print("Memory estimate scaling with batch size (FSDP, 7B params, bf16):")
    print(f"{'Batch Size':>12} {'Memory/GPU (GB)':>16}")
    print("-" * 30)

    fsdp_config = DistributedConfig(
        strategy=DistributedStrategy.FSDP,
        num_gpus=8,
        mixed_precision=MixedPrecision.BF16,
        gradient_checkpointing=True,
    )
    for bs in [1, 2, 4, 8, 16]:
        mem = fsdp_config.estimate_memory_gb(7_000, bs, sequence_length=2048)
        print(f"{bs:>12} {mem:>16.1f}")


if __name__ == "__main__":
    main()
