from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from reid.models.feature_enhancers import (
    ECAChannelAttention,
    IdentityEnhancer,
    StableResidualAdapter,
    build_feature_enhancer,
)


def _make_cfg(
    name="none",
    channels=8,
    res_scale_init=0.0,
    learnable=True,
    res_scale_max=0.1,
    **eca_kwargs
):
    eca_defaults = {"K_SIZE": 3}
    eca_defaults.update(eca_kwargs)
    feature_enhancer = SimpleNamespace(
        NAME=name,
        CHANNELS=channels,
        RES_SCALE_INIT=res_scale_init,
        RES_SCALE_LEARNABLE=learnable,
        RES_SCALE_MAX=res_scale_max,
        ECA=SimpleNamespace(**eca_defaults),
    )
    return SimpleNamespace(MODEL=SimpleNamespace(FEATURE_ENHANCER=feature_enhancer))


def test_identity_enhancer_returns_input_unchanged():
    x = torch.randn(2, 8, 8, 4)
    y = IdentityEnhancer()(x)

    assert y is x
    assert torch.equal(y, x)


def test_eca_channel_attention_preserves_shape():
    x = torch.randn(2, 8, 8, 4)
    module = ECAChannelAttention(channels=8, k_size=3)

    y = module(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("k_size", [0, 2, -1])
def test_eca_channel_attention_rejects_invalid_kernel_size(k_size):
    with pytest.raises(ValueError):
        ECAChannelAttention(channels=8, k_size=k_size)


def test_stable_residual_zero_scale_matches_input():
    class DoubleEnhancer(nn.Module):
        def forward(self, x):
            return x * 2.0

    x = torch.randn(2, 8, 8, 4)
    module = StableResidualAdapter(
        DoubleEnhancer(),
        res_scale_init=0.0,
        learnable=False,
        res_scale_max=0.1,
    )

    y = module(x)

    assert torch.allclose(y, x)


def test_stable_residual_clamps_large_scale_for_safe_adaptation():
    class DoubleEnhancer(nn.Module):
        def forward(self, x):
            return x * 2.0

    x = torch.randn(2, 8, 8, 4)
    module = StableResidualAdapter(
        DoubleEnhancer(),
        res_scale_init=1.0,
        learnable=False,
        res_scale_max=0.1,
    )

    y = module(x)

    assert torch.allclose(y, x * 1.1)


def test_build_feature_enhancer_supports_none_and_eca_safe():
    none_module = build_feature_enhancer(_make_cfg(name="none"), channels=8)
    assert isinstance(none_module, IdentityEnhancer)

    eca_module = build_feature_enhancer(_make_cfg(name="eca_safe"), channels=8)
    assert isinstance(eca_module, StableResidualAdapter)
    assert isinstance(eca_module.enhancer, ECAChannelAttention)

