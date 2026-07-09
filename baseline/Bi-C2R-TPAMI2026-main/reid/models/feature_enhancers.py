import torch
import torch.nn as nn


class IdentityEnhancer(nn.Module):
    """No-op enhancer for the original baseline."""

    def forward(self, x):
        return x

    def get_scale_state(self):
        return None


class ECAChannelAttention(nn.Module):
    """Efficient channel attention using only global average statistics."""

    def __init__(self, channels, k_size=3):
        super(ECAChannelAttention, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive, got {}".format(channels))
        if k_size <= 0 or k_size % 2 == 0:
            raise ValueError("k_size must be a positive odd integer, got {}".format(k_size))

        self.channels = int(channels)
        self.k_size = int(k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.k_size,
            padding=(self.k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D feature map [B, C, H, W], got {}".format(tuple(x.shape)))
        if x.size(1) != self.channels:
            raise ValueError("expected {} channels, got {}".format(self.channels, x.size(1)))

        weights = self.avg_pool(x)
        weights = self.conv(weights.squeeze(-1).transpose(-1, -2))
        weights = weights.transpose(-1, -2).unsqueeze(-1)
        weights = self.sigmoid(weights)
        return x * weights.expand_as(x)


class StableResidualAdapter(nn.Module):
    """Wrap an enhancer with bounded residual strength.

    The default zero scale keeps the initial model behavior identical to the
    original baseline, which is important for RFL old-gallery compatibility.
    """

    def __init__(self, enhancer, res_scale_init=0.0, learnable=True, res_scale_max=0.1):
        super(StableResidualAdapter, self).__init__()
        self.enhancer = enhancer
        self.res_scale_max = None if res_scale_max is None else float(res_scale_max)
        scale = torch.tensor(float(res_scale_init))
        if learnable:
            self.res_scale = nn.Parameter(scale)
        else:
            self.register_buffer("res_scale", scale)

    def _bounded_scale_value(self):
        raw = float(self.res_scale.detach().cpu().item())
        effective = raw
        if self.res_scale_max is not None and self.res_scale_max >= 0:
            effective = min(max(raw, 0.0), float(self.res_scale_max))
        return raw, effective

    def _bounded_scale(self, x):
        scale = self.res_scale.to(device=x.device, dtype=x.dtype)
        if self.res_scale_max is not None and self.res_scale_max >= 0:
            scale = torch.clamp(scale, min=0.0, max=float(self.res_scale_max))
        return scale

    def get_scale_state(self):
        raw, effective = self._bounded_scale_value()
        return {
            "raw": raw,
            "effective": effective,
            "max": self.res_scale_max,
            "learnable": isinstance(self.res_scale, nn.Parameter),
        }

    def forward(self, x):
        enhanced = self.enhancer(x)
        scale = self._bounded_scale(x)
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


def _cfg_float_or_none(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    return float(value)


def build_feature_enhancer(cfg, channels=2048):
    feature_cfg = _cfg_get(_cfg_get(cfg, "MODEL"), "FEATURE_ENHANCER")
    if feature_cfg is None:
        return IdentityEnhancer()

    name = str(_cfg_get(feature_cfg, "NAME", "none")).lower()
    if name in ("none", "identity", ""):
        return IdentityEnhancer()
    if name not in ("eca", "eca_safe"):
        raise ValueError("unsupported feature enhancer: {}".format(name))

    configured_channels = int(_cfg_get(feature_cfg, "CHANNELS", channels))
    if configured_channels != int(channels):
        raise ValueError(
            "FEATURE_ENHANCER.CHANNELS ({}) does not match backbone channels ({})".format(
                configured_channels, channels
            )
        )

    eca_cfg = _cfg_get(feature_cfg, "ECA")
    eca = ECAChannelAttention(
        channels=channels,
        k_size=int(_cfg_get(eca_cfg, "K_SIZE", 3)),
    )
    return StableResidualAdapter(
        eca,
        res_scale_init=float(_cfg_get(feature_cfg, "RES_SCALE_INIT", 0.0)),
        learnable=_cfg_bool(_cfg_get(feature_cfg, "RES_SCALE_LEARNABLE", True)),
        res_scale_max=_cfg_float_or_none(_cfg_get(feature_cfg, "RES_SCALE_MAX", 0.1), default=0.1),
    )


__all__ = [
    "IdentityEnhancer",
    "ECAChannelAttention",
    "StableResidualAdapter",
    "build_feature_enhancer",
]
