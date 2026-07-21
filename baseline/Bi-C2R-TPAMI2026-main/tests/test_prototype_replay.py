from collections import OrderedDict

import torch

from reid.utils.prototype_replay import (
    PrototypeBank,
    compute_opcr_loss,
)


def _save_gallery(path, features, labels):
    torch.save(
        {
            "features": OrderedDict(features),
            "labels": OrderedDict(labels),
        },
        path,
    )


def test_prototype_bank_builds_normalized_domain_aware_prototypes(tmp_path):
    gallery_a = tmp_path / "market1501_features.pth.tar"
    gallery_b = tmp_path / "dukemtmc_features.pth.tar"

    _save_gallery(
        gallery_a,
        [("a1", torch.tensor([3.0, 0.0, 0.0, 0.0]))],
        [("a1", 7)],
    )
    _save_gallery(
        gallery_b,
        [("b1", torch.tensor([0.0, 4.0, 0.0, 0.0]))],
        [("b1", 7)],
    )

    bank = PrototypeBank.from_gallery_files(
        [str(gallery_a), str(gallery_b)],
        prototypes_per_id=1,
    )

    assert len(bank) == 2
    assert bank.features.shape == (2, 4)
    assert torch.allclose(bank.features.norm(dim=1), torch.ones(2))
    assert bank.identity_ids.unique().numel() == 2


def test_prototype_bank_skips_gallery_without_labels(tmp_path):
    gallery = tmp_path / "legacy_features.pth.tar"
    torch.save(
        {"features": OrderedDict([("x", torch.ones(4))])},
        gallery,
    )

    bank = PrototypeBank.from_gallery_files([str(gallery)])

    assert len(bank) == 0


def test_opcr_identity_mapping_has_near_zero_loss():
    old_features = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)

    total, pieces = compute_opcr_loss(
        old_features,
        old_features,
        old_features,
        temperature=0.05,
        weight_relation=0.5,
        weight_cycle=0.1,
        weight_separation=0.0,
    )

    assert torch.isfinite(total)
    assert total.item() < 1e-5
    assert pieces["relation"].item() < 1e-5
    assert pieces["cycle"].item() < 1e-5


def test_opcr_loss_is_finite_for_single_prototype():
    old_features = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    transformed = old_features + 0.1
    cycled = transformed - 0.1

    total, pieces = compute_opcr_loss(
        old_features,
        transformed,
        cycled,
        temperature=0.05,
        weight_relation=0.5,
        weight_cycle=0.1,
        weight_separation=0.2,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(pieces["relation"])
    assert torch.isfinite(pieces["cycle"])
    assert torch.isfinite(pieces["separation"])


def test_prototype_bank_prefers_gallery_keys_when_available(tmp_path):
    gallery = tmp_path / "market1501_features.pth.tar"
    torch.save(
        {
            "features": OrderedDict(
                [
                    ("query_img", torch.tensor([1.0, 0.0, 0.0, 0.0])),
                    ("gallery_img", torch.tensor([0.0, 1.0, 0.0, 0.0])),
                ]
            ),
            "labels": OrderedDict([("query_img", 1), ("gallery_img", 2)]),
            "gallery_keys": ["gallery_img"],
            "query_keys": ["query_img"],
        },
        gallery,
    )

    bank = PrototypeBank.from_gallery_files([str(gallery)])

    assert len(bank) == 1
    assert bank.meta[0]["label"] == 2
    assert torch.allclose(bank.features[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
