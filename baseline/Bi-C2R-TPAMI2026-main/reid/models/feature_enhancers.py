import torch
import torch.nn as nn


class IdentityEnhancer(nn.Module):
    """No-op enhancer for the original baseline."""

    def forward(self, x):
        return x


class HPAEnhancer(nn.Module):
    """Hybrid pooling attention with ablation switches.

    The default setting is HPA-Safe: average-pooling branch enabled and
    max-pooling branch disabled. This keeps the module conservative for
    lifelong ReID where background responses can be unstable across tasks.
    """

    def __init__(
        self,
        channels,
        factor=32,
        use_avg_pool=True,
        use_max_pool=False,
        use_group_norm=True,
    ):
        super(HPAEnhancer, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive, got {}".format(channels))
        if factor <= 0:
            raise ValueError("factor must be positive, got {}".format(factor))
        if channels % factor != 0:
            raise ValueError("channels ({}) must be divisible by factor ({})".format(channels, factor))
        if not use_avg_pool and not use_max_pool:
            raise ValueError("at least one HPA pooling branch must be enabled")

        self.channels = int(channels)
        self.groups = int(factor)
        self.group_channels = self.channels // self.groups
        self.use_avg_pool = bool(use_avg_pool)
        self.use_max_pool = bool(use_max_pool)
        self.use_group_norm = bool(use_group_norm)

        self.softmax = nn.Softmax(dim=-1)
        self.conv1x1 = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        if self.use_group_norm:
            self.norm = nn.GroupNorm(self.group_channels, self.group_channels)
        else:
            self.norm = nn.Identity()

    def _pool_hw(self, x, mode):
        if mode == "avg":
            pooled_h = x.mean(dim=3, keepdim=True)
            pooled_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        elif mode == "max":
            pooled_h = x.max(dim=3, keepdim=True)[0]
            pooled_w = x.max(dim=2, keepdim=True)[0].permute(0, 1, 3, 2)
        else:
            raise ValueError("unsupported pooling mode: {}".format(mode))
        return pooled_h, pooled_w

    def _branch(self, group_x, mode):
        _, _, height, width = group_x.shape
        pooled_h, pooled_w = self._pool_hw(group_x, mode)
        pooled = self.conv1x1(torch.cat([pooled_h, pooled_w], dim=2))
        gate_h, gate_w = torch.split(pooled, [height, width], dim=2)
        gated = group_x * gate_h.sigmoid() * gate_w.permute(0, 1, 3, 2).sigmoid()
        return self.norm(gated)

    def _descriptor(self, feat, mode):
        if mode == "max":
            desc = feat.amax(dim=(2, 3), keepdim=True)
        else:
            desc = feat.mean(dim=(2, 3), keepdim=True)
        return self.softmax(desc.reshape(feat.size(0), -1, 1).permute(0, 2, 1))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D feature map [B, C, H, W], got {}".format(tuple(x.shape)))
        if x.size(1) != self.channels:
            raise ValueError("expected {} channels, got {}".format(self.channels, x.size(1)))

        batch, channels, height, width = x.shape
        group_x = x.reshape(batch * self.groups, self.group_channels, height, width)

        branches = []
        if self.use_avg_pool:
            avg_feat = self._branch(group_x, "avg")
            branches.append((avg_feat, self._descriptor(avg_feat, "avg")))
        if self.use_max_pool:
            max_feat = self._branch(group_x, "max")
            branches.append((max_feat, self._descriptor(max_feat, "max")))

        if len(branches) == 1:
            feat, desc = branches[0]
            weights = torch.matmul(desc, feat.reshape(batch * self.groups, self.group_channels, -1))
        else:
            avg_feat, avg_desc = branches[0]
            max_feat, max_desc = branches[1]
            avg_flat = avg_feat.reshape(batch * self.groups, self.group_channels, -1)
            max_flat = max_feat.reshape(batch * self.groups, self.group_channels, -1)
            weights = torch.matmul(avg_desc, max_flat) + torch.matmul(max_desc, avg_flat)

        weights = weights.reshape(batch * self.groups, 1, height, width).sigmoid()
        return (group_x * weights).reshape(batch, channels, height, width)


class StableResidualAdapter(nn.Module):
    """Wrap an enhancer with a controllable residual path."""

    def __init__(self, enhancer, res_scale_init=0.0, learnable=True):
        super(StableResidualAdapter, self).__init__()
        self.enhancer = enhancer
        scale = torch.tensor(float(res_scale_init))
        if learnable:
            self.res_scale = nn.Parameter(scale)
        else:
            self.register_buffer("res_scale", scale)

    def forward(self, x):
        enhanced = self.enhancer(x)
        scale = self.res_scale.to(device=x.device, dtype=x.dtype)
        return x + scale * (enhanced - x)


def _cfg_get(node, name, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _cfg_bool(value):
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def build_feature_enhancer(cfg, channels=2048):
    feature_cfg = _cfg_get(_cfg_get(cfg, "MODEL"), "FEATURE_ENHANCER")
    if feature_cfg is None:
        return IdentityEnhancer()

    name = str(_cfg_get(feature_cfg, "NAME", "none")).lower()
    if name in ("none", "identity", ""):
        return IdentityEnhancer()
    if name != "hpa":
        raise ValueError("unsupported feature enhancer: {}".format(name))

    configured_channels = int(_cfg_get(feature_cfg, "CHANNELS", channels))
    if configured_channels != int(channels):
        raise ValueError(
            "FEATURE_ENHANCER.CHANNELS ({}) does not match backbone channels ({})".format(
                configured_channels, channels
            )
        )

    hpa_cfg = _cfg_get(feature_cfg, "HPA")
    hpa = HPAEnhancer(
        channels=channels,
        factor=int(_cfg_get(hpa_cfg, "FACTOR", 32)),
        use_avg_pool=_cfg_bool(_cfg_get(hpa_cfg, "USE_AVG_POOL", True)),
        use_max_pool=_cfg_bool(_cfg_get(hpa_cfg, "USE_MAX_POOL", False)),
        use_group_norm=_cfg_bool(_cfg_get(hpa_cfg, "USE_GROUP_NORM", True)),
    )
    return StableResidualAdapter(
        hpa,
        res_scale_init=float(_cfg_get(feature_cfg, "RES_SCALE_INIT", 0.0)),
        learnable=_cfg_bool(_cfg_get(feature_cfg, "RES_SCALE_LEARNABLE", True)),
    )


__all__ = [
    "IdentityEnhancer",
    "HPAEnhancer",
    "StableResidualAdapter",
    "build_feature_enhancer",
]
