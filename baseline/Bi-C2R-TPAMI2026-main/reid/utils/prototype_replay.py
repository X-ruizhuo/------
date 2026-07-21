from __future__ import absolute_import

from collections import OrderedDict, defaultdict
import os

import torch
import torch.nn.functional as F


def _to_int(value):
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _as_feature(value):
    if torch.is_tensor(value):
        feature = value.detach().float().view(-1)
    else:
        feature = torch.tensor(value, dtype=torch.float32).view(-1)
    return F.normalize(feature.unsqueeze(0), p=2, dim=1).squeeze(0)


def _domain_name(path, domain_index):
    name = os.path.basename(path)
    suffix = "_features.pth.tar"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name or "domain_{}".format(domain_index)


class PrototypeBank(object):
    """Compact feature-only replay bank for old ReID identities."""

    def __init__(self, features=None, identity_ids=None, domain_ids=None, meta=None):
        if features is None:
            features = torch.empty(0, 0, dtype=torch.float32)
        if identity_ids is None:
            identity_ids = torch.empty(0, dtype=torch.long)
        if domain_ids is None:
            domain_ids = torch.empty(0, dtype=torch.long)

        self.features = F.normalize(features.float(), p=2, dim=1) if features.numel() > 0 else features.float()
        self.identity_ids = identity_ids.long()
        self.domain_ids = domain_ids.long()
        self.meta = meta or []

    def __len__(self):
        return int(self.features.size(0))

    @property
    def is_empty(self):
        return len(self) == 0

    @classmethod
    def empty(cls, feat_dim=2048):
        return cls(torch.empty(0, feat_dim), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    @classmethod
    def from_gallery_files(cls, paths, prototypes_per_id=1):
        groups = OrderedDict()
        domain_names = []
        prototypes_per_id = max(1, int(prototypes_per_id))

        for domain_index, path in enumerate(paths):
            if not path or not os.path.exists(path):
                continue
            payload = torch.load(path, map_location="cpu")
            features = payload.get("features", {})
            labels = payload.get("labels")
            if labels is None:
                continue

            domain = _domain_name(path, domain_index)
            domain_names.append(domain)
            replay_keys = payload.get("gallery_keys")
            if replay_keys is None:
                replay_keys = list(features.keys())

            for key in replay_keys:
                if key not in features:
                    continue
                if key not in labels:
                    continue
                feature = features[key]
                label = _to_int(labels[key])
                group_key = (domain_index, domain, label)
                groups.setdefault(group_key, []).append(_as_feature(feature))

        prototype_features = []
        prototype_identity_ids = []
        prototype_domain_ids = []
        meta = []

        for identity_index, ((domain_index, domain, label), values) in enumerate(groups.items()):
            if not values:
                continue
            chunks = _split_feature_chunks(values, prototypes_per_id)
            for proto_index, chunk in enumerate(chunks):
                stacked = torch.stack(chunk, dim=0)
                prototype = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
                prototype_features.append(prototype)
                prototype_identity_ids.append(identity_index)
                prototype_domain_ids.append(domain_index)
                meta.append(
                    {
                        "domain": domain,
                        "label": label,
                        "prototype_index": proto_index,
                        "num_features": len(chunk),
                    }
                )

        if not prototype_features:
            return cls.empty()

        return cls(
            torch.stack(prototype_features, dim=0),
            torch.tensor(prototype_identity_ids, dtype=torch.long),
            torch.tensor(prototype_domain_ids, dtype=torch.long),
            meta=meta,
        )

    def sample(self, sample_size, device=None, balance_by_domain=True):
        if self.is_empty:
            return None

        sample_size = max(1, int(sample_size))
        if balance_by_domain and self.domain_ids.numel() > 0:
            indices = self._sample_balanced(sample_size)
        else:
            indices = self._sample_uniform(sample_size)

        features = self.features[indices]
        identity_ids = self.identity_ids[indices]
        domain_ids = self.domain_ids[indices]
        if device is not None:
            features = features.to(device)
            identity_ids = identity_ids.to(device)
            domain_ids = domain_ids.to(device)
        return {
            "features": features,
            "identity_ids": identity_ids,
            "domain_ids": domain_ids,
        }

    def _sample_uniform(self, sample_size):
        if len(self) >= sample_size:
            return torch.randperm(len(self))[:sample_size]
        return torch.randint(0, len(self), (sample_size,), dtype=torch.long)

    def _sample_balanced(self, sample_size):
        by_domain = defaultdict(list)
        for index, domain_id in enumerate(self.domain_ids.tolist()):
            by_domain[domain_id].append(index)

        domains = sorted(by_domain.keys())
        if not domains:
            return self._sample_uniform(sample_size)

        selected = []
        per_domain = max(1, sample_size // len(domains))
        for domain_id in domains:
            domain_indices = torch.tensor(by_domain[domain_id], dtype=torch.long)
            if domain_indices.numel() >= per_domain:
                take = domain_indices[torch.randperm(domain_indices.numel())[:per_domain]]
            else:
                take = domain_indices[torch.randint(0, domain_indices.numel(), (per_domain,), dtype=torch.long)]
            selected.append(take)

        indices = torch.cat(selected, dim=0)
        if indices.numel() < sample_size:
            extra = self._sample_uniform(sample_size - indices.numel())
            indices = torch.cat([indices, extra], dim=0)
        elif indices.numel() > sample_size:
            indices = indices[torch.randperm(indices.numel())[:sample_size]]
        return indices


def _split_feature_chunks(values, num_chunks):
    if len(values) <= num_chunks:
        return [[value] for value in values]

    chunks = [[] for _ in range(num_chunks)]
    for index, value in enumerate(values):
        chunks[index % num_chunks].append(value)
    return [chunk for chunk in chunks if chunk]


def _masked_relation_distribution(features, temperature, log_prob):
    n = features.size(0)
    logits = torch.mm(features, features.t()) / max(float(temperature), 1e-6)
    if n > 1:
        eye = torch.eye(n, dtype=torch.bool, device=features.device)
        logits = logits.masked_fill(eye, -1e4)
    if log_prob:
        return F.log_softmax(logits, dim=1)
    return F.softmax(logits, dim=1)


def _separation_loss(features, identity_ids=None, margin=0.2):
    n = features.size(0)
    if n <= 1:
        return features.new_zeros(())

    sim = torch.mm(features, features.t())
    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    neg_mask = ~eye
    if identity_ids is not None:
        identity_ids = identity_ids.view(-1)
        same = identity_ids.unsqueeze(0).eq(identity_ids.unsqueeze(1))
        neg_mask = neg_mask & ~same
    if neg_mask.sum() == 0:
        return features.new_zeros(())
    return F.relu(sim[neg_mask] - float(margin)).mean()


def compute_opcr_loss(
    old_features,
    transformed_features,
    cycled_features,
    identity_ids=None,
    temperature=0.05,
    weight_relation=0.5,
    weight_cycle=0.1,
    weight_separation=0.0,
    separation_margin=0.2,
):
    old_features = F.normalize(old_features.float(), p=2, dim=1)
    transformed_features = F.normalize(transformed_features.float(), p=2, dim=1)
    cycled_features = F.normalize(cycled_features.float(), p=2, dim=1)

    if old_features.size(0) > 1:
        old_relation = _masked_relation_distribution(old_features.detach(), temperature, log_prob=False)
        new_relation = _masked_relation_distribution(transformed_features, temperature, log_prob=True)
        relation = F.kl_div(new_relation, old_relation, reduction="batchmean")
    else:
        relation = old_features.new_zeros(())

    cycle = (1.0 - F.cosine_similarity(cycled_features, old_features.detach(), dim=1)).mean()
    separation = _separation_loss(transformed_features, identity_ids=identity_ids, margin=separation_margin)
    total = (
        float(weight_relation) * relation
        + float(weight_cycle) * cycle
        + float(weight_separation) * separation
    )
    return total, {
        "relation": relation.detach(),
        "cycle": cycle.detach(),
        "separation": separation.detach(),
    }
