"""
Tests for pipeline.distributed_config: DistributedConfig, DistributedStrategy, etc.
"""

import pytest
from pipeline.distributed_config import (
    DistributedConfig,
    DistributedStrategy,
    Backend,
    MixedPrecision,
)


def test_default_config_is_valid():
    """Default config should pass validation with no errors."""
    config = DistributedConfig()
    errors = config.validate()
    assert errors == [], f"Default config has errors: {errors}"
    assert config.is_valid()


def test_estimate_memory_returns_positive_float():
    """Memory estimate should always be a positive float."""
    config = DistributedConfig(num_gpus=4, strategy=DistributedStrategy.DDP)
    mem = config.estimate_memory_gb(model_params_millions=1_000, batch_size=4)
    assert isinstance(mem, float)
    assert mem > 0.0


def test_memory_scales_with_batch_size():
    """Larger batch sizes should require more memory (activations grow with batch)."""
    config = DistributedConfig(num_gpus=1, strategy=DistributedStrategy.DDP)
    mem_small = config.estimate_memory_gb(1_000, batch_size=1)
    mem_large = config.estimate_memory_gb(1_000, batch_size=32)
    assert mem_large > mem_small, f"Expected mem_large > mem_small: {mem_large} vs {mem_small}"


def test_incompatible_settings_caught_by_validate():
    """DataParallel + multi-node should fail validation."""
    config = DistributedConfig(
        strategy=DistributedStrategy.DATA_PARALLEL,
        num_gpus=4,
        num_nodes=2,  # Invalid for DataParallel
    )
    errors = config.validate()
    assert len(errors) > 0
    assert not config.is_valid()


def test_to_torch_config_returns_dict_with_required_keys():
    """to_torch_config() must return a dict with backend, world_size, init_method."""
    config = DistributedConfig(num_gpus=2, strategy=DistributedStrategy.DDP)
    torch_cfg = config.to_torch_config()
    assert isinstance(torch_cfg, dict)
    for key in ("backend", "world_size", "init_method"):
        assert key in torch_cfg, f"Missing key: {key}"
    assert torch_cfg["world_size"] == 2


def test_fsdp_config_has_correct_backend():
    """FSDP should require nccl and produce an fsdp_config sub-dict."""
    config = DistributedConfig(
        strategy=DistributedStrategy.FSDP,
        num_gpus=8,
        backend=Backend.NCCL,
        mixed_precision=MixedPrecision.BF16,
    )
    assert config.is_valid()
    torch_cfg = config.to_torch_config()
    assert "fsdp_config" in torch_cfg
    assert torch_cfg["backend"] == "nccl"
