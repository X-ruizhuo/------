from types import SimpleNamespace
import math

import torch

from reid.models.feature_enhancers import (
    GRNSafeEnhancer,
    IdentityEnhancer,
    StableResidualAdapter,
    build_feature_enhancer,
)


def _cfg(name="grn", scale=0.05, learnable=True, use_bias=True):
    grn_cfg = SimpleNamespace(USE_BIAS=use_bias, EPS=1e-6)
    enhancer_cfg = SimpleNamespace(
        NAME=name,
        CHANNELS=8,
        RES_SCALE_INIT=scale,
        RES_SCALE_LEARNABLE=learnable,
        GRN=grn_cfg,
    )
    return SimpleNamespace(MODEL=SimpleNamespace(FEATURE_ENHANCER=enhancer_cfg))


def test_identity_enhancer_is_exact_identity():
    x = torch.randn(2, 8, 4, 4)
    enhancer = IdentityEnhancer()

    y = enhancer(x)

    assert y is x
    assert list(enhancer.parameters()) == []


def test_grn_safe_enhancer_preserves_shape_and_starts_as_identity():
    x = torch.randn(2, 8, 4, 4)
    enhancer = build_feature_enhancer(_cfg(), channels=8)

    y = enhancer(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=0.0, rtol=0.0)
    assert math.isclose(enhancer.get_scale_state()["effective"], 0.05, rel_tol=0.0, abs_tol=1e-8)


def test_residual_scale_zero_blocks_a_non_identity_inner_module():
    x = torch.randn(2, 8, 4, 4)
    inner = GRNSafeEnhancer(8, use_bias=True, eps=1e-6)
    with torch.no_grad():
        inner.gamma.fill_(0.5)
        inner.beta.fill_(0.25)
    enhancer = StableResidualAdapter(inner, scale_init=0.0, learnable=False)

    y = enhancer(x)

    assert torch.allclose(y, x, atol=0.0, rtol=0.0)


def test_grn_without_beta_forwards():
    x = torch.randn(2, 8, 4, 4)
    enhancer = build_feature_enhancer(_cfg(use_bias=False), channels=8)

    y = enhancer(x)

    assert y.shape == x.shape
