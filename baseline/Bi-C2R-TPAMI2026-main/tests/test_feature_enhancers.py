from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from reid.models.feature_enhancers import (
    FUSEChannelAttention,
    IdentityEnhancer,
    StableResidualAdapter,
    build_feature_enhancer,
)


def _make_cfg(name="none", channels=8, res_scale_init=0.0, learnable=True, **fuse_kwargs):
    fuse_defaults = {
        "REDUCTION": 4,
        "LF_RATIO": 0.25,
        "EPS": 1e-6,
        "USE_STD": True,
        "USE_FREQ": True,
        "USE_TAU": True,
        "USE_PER_CHANNEL_WEIGHT": True,
    }
    fuse_defaults.update(fuse_kwargs)
    feature_enhancer = SimpleNamespace(
        NAME=name,
        CHANNELS=channels,
        RES_SCALE_INIT=res_scale_init,
        RES_SCALE_LEARNABLE=learnable,
        FUSE=SimpleNamespace(**fuse_defaults),
    )
    return SimpleNamespace(MODEL=SimpleNamespace(FEATURE_ENHANCER=feature_enhancer))


def test_identity_enhancer_returns_input_unchanged():
    x = torch.randn(2, 8, 8, 4)
    y = IdentityEnhancer()(x)
    assert y is x
    assert torch.equal(y, x)


def test_fuse_channel_attention_preserves_shape():
    x = torch.randn(2, 8, 8, 4)
    module = FUSEChannelAttention(channels=8, reduction=4, lf_ratio=0.25)
    y = module(x)
    assert y.shape == x.shape


def test_stable_residual_zero_scale_matches_input():
    class DoubleEnhancer(nn.Module):
        def forward(self, x):
            return x * 2.0

    x = torch.randn(2, 8, 8, 4)
    module = StableResidualAdapter(DoubleEnhancer(), res_scale_init=0.0, learnable=False)
    y = module(x)
    assert torch.allclose(y, x)


@pytest.mark.parametrize(
    "overrides",
    [
        {"USE_FREQ": False},
        {"USE_TAU": False},
        {"USE_STD": False},
        {"USE_PER_CHANNEL_WEIGHT": False},
        {"USE_FREQ": False, "USE_TAU": False, "USE_STD": False},
    ],
)
def test_fuse_ablation_switches_forward(overrides):
    arg_names = {
        "USE_STD": "use_std",
        "USE_FREQ": "use_freq",
        "USE_TAU": "use_tau",
        "USE_PER_CHANNEL_WEIGHT": "use_per_channel_weight",
    }
    kwargs = {arg_names[key]: value for key, value in overrides.items()}
    x = torch.randn(2, 8, 8, 4)
    module = FUSEChannelAttention(channels=8, reduction=4, lf_ratio=0.25, **kwargs)
    y = module(x)
    assert y.shape == x.shape


def test_build_feature_enhancer_supports_none_and_fuse():
    none_module = build_feature_enhancer(_make_cfg(name="none"), channels=8)
    assert isinstance(none_module, IdentityEnhancer)

    fuse_module = build_feature_enhancer(_make_cfg(name="fuse"), channels=8)
    assert isinstance(fuse_module, StableResidualAdapter)


def test_make_model_accepts_cfg_and_preserves_training_outputs():
    from reid.models.resnet import make_model

    cfg = _make_cfg(name="none", channels=2048)
    model = make_model(cfg, num_class=10, camera_num=0, view_num=0, pretrain=False)
    model.train()

    x = torch.randn(2, 3, 256, 128)
    global_feat, bn_feat, cls_outputs, feat_final_layer = model(x)

    assert list(global_feat.shape) == [2, 2048]
    assert list(bn_feat.shape) == [2, 2048]
    assert list(cls_outputs.shape) == [2, 10]
    assert feat_final_layer.shape[1] == 2048
