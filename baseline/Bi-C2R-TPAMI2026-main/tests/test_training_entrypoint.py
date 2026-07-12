from types import SimpleNamespace

import torch

from reid.models.resnet import make_model


def _cfg(name="grn", scale=0.05, learnable=True, use_bias=True):
    grn_cfg = SimpleNamespace(USE_BIAS=use_bias, EPS=1e-6)
    enhancer_cfg = SimpleNamespace(
        NAME=name,
        CHANNELS=2048,
        RES_SCALE_INIT=scale,
        RES_SCALE_LEARNABLE=learnable,
        GRN=grn_cfg,
    )
    return SimpleNamespace(MODEL=SimpleNamespace(FEATURE_ENHANCER=enhancer_cfg))


def test_make_model_accepts_cfg_and_keeps_training_output_shapes():
    model = make_model(_cfg(), num_class=10, camera_num=0, view_num=0, pretrain=False)
    model.train()
    x = torch.randn(2, 3, 64, 32)

    global_feat, bn_feat, cls_outputs, feat_final_layer = model(x)

    assert global_feat.shape == (2, 2048)
    assert bn_feat.shape == (2, 2048)
    assert cls_outputs.shape == (2, 10)
    assert feat_final_layer.shape[0] == 2
    assert feat_final_layer.shape[1] == 2048


def test_make_model_keeps_old_positional_call_working():
    model = make_model(10, 0, 0, False)

    assert model.classifier.out_features == 10
