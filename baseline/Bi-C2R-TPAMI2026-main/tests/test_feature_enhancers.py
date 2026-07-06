from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from reid.models.feature_enhancers import (
    HPAEnhancer,
    IdentityEnhancer,
    StableResidualAdapter,
    build_feature_enhancer,
)


def _make_cfg(name="none", channels=8, res_scale_init=0.0, learnable=True, **hpa_kwargs):
    hpa_defaults = {
        "FACTOR": 4,
        "USE_AVG_POOL": True,
        "USE_MAX_POOL": False,
        "USE_GROUP_NORM": True,
    }
    hpa_defaults.update(hpa_kwargs)
    feature_enhancer = SimpleNamespace(
        NAME=name,
        CHANNELS=channels,
        RES_SCALE_INIT=res_scale_init,
        RES_SCALE_LEARNABLE=learnable,
        HPA=SimpleNamespace(**hpa_defaults),
    )
    return SimpleNamespace(MODEL=SimpleNamespace(FEATURE_ENHANCER=feature_enhancer))


def test_identity_enhancer_returns_input_unchanged():
    x = torch.randn(2, 8, 8, 4)
    y = IdentityEnhancer()(x)
    assert y is x
    assert torch.equal(y, x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_avg_pool": True, "use_max_pool": False},
        {"use_avg_pool": True, "use_max_pool": True},
        {"use_avg_pool": False, "use_max_pool": True},
        {"use_avg_pool": True, "use_max_pool": False, "use_group_norm": False},
    ],
)
def test_hpa_enhancer_ablation_switches_preserve_shape(kwargs):
    x = torch.randn(2, 8, 8, 4)
    module = HPAEnhancer(channels=8, factor=4, **kwargs)
    y = module(x)
    assert y.shape == x.shape


def test_hpa_enhancer_rejects_disabling_all_pooling_branches():
    with pytest.raises(ValueError):
        HPAEnhancer(channels=8, factor=4, use_avg_pool=False, use_max_pool=False)


def test_stable_residual_zero_scale_matches_input():
    class DoubleEnhancer(nn.Module):
        def forward(self, x):
            return x * 2.0

    x = torch.randn(2, 8, 8, 4)
    module = StableResidualAdapter(DoubleEnhancer(), res_scale_init=0.0, learnable=False)
    y = module(x)
    assert torch.allclose(y, x)


def test_build_feature_enhancer_supports_none_and_hpa():
    none_module = build_feature_enhancer(_make_cfg(name="none"), channels=8)
    assert isinstance(none_module, IdentityEnhancer)

    hpa_module = build_feature_enhancer(_make_cfg(name="hpa"), channels=8)
    assert isinstance(hpa_module, StableResidualAdapter)
    assert isinstance(hpa_module.enhancer, HPAEnhancer)
