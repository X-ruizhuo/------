import torch
import torch.nn as nn


def _cfg_get(node, name, default=None):
    if node is None:
        return default
    if hasattr(node, name):
        return getattr(node, name)
    if isinstance(node, dict):
        return node.get(name, default)
    return default


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


class IdentityEnhancer(nn.Module):
    def forward(self, x):
        return x


class GRNSafeEnhancer(nn.Module):
    """Global Response Normalization for NCHW feature maps.

    Parameters are zero-initialized, so the module starts as an exact identity.
    """

    def __init__(self, channels, use_bias=True, eps=1e-6, detach_response=False):
        super(GRNSafeEnhancer, self).__init__()
        self.channels = int(channels)
        self.use_bias = _as_bool(use_bias)
        self.eps = float(eps)
        self.detach_response = _as_bool(detach_response)

        self.gamma = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        else:
            self.register_parameter("beta", None)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("GRNSafeEnhancer expects a 4D tensor [N, C, H, W].")
        if x.size(1) != self.channels:
            raise ValueError(
                "GRNSafeEnhancer expected {} channels, got {}.".format(
                    self.channels, x.size(1)
                )
            )

        response = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        response = response / (response.mean(dim=1, keepdim=True) + self.eps)
        if self.detach_response:
            response = response.detach()

        out = (1.0 + self.gamma * response) * x
        if self.beta is not None:
            out = out + self.beta
        return out

    def extra_repr(self):
        return "channels={}, use_bias={}, eps={}, detach_response={}".format(
            self.channels, self.use_bias, self.eps, self.detach_response
        )


class StableResidualAdapter(nn.Module):
    """Wrap an enhancer with a bounded residual strength."""

    def __init__(self, enhancer, scale_init=0.0, learnable=True, scale_max=None):
        super(StableResidualAdapter, self).__init__()
        self.enhancer = enhancer
        self.scale_learnable = _as_bool(learnable)
        self.scale_max = None if scale_max is None else float(scale_max)
        if self.scale_max is not None and self.scale_max <= 0:
            self.scale_max = None

        scale = torch.tensor(float(scale_init), dtype=torch.float32)
        if self.scale_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def _effective_scale(self):
        scale = self.residual_scale
        if self.scale_max is not None:
            scale = torch.clamp(scale, min=0.0, max=self.scale_max)
        return scale

    def forward(self, x):
        enhanced = self.enhancer(x)
        scale = self._effective_scale().to(device=x.device, dtype=x.dtype)
        return x + scale * (enhanced - x)

    def get_scale_state(self):
        raw = float(self.residual_scale.detach().cpu().item())
        effective = float(self._effective_scale().detach().cpu().item())
        return {
            "raw": raw,
            "effective": effective,
            "max": self.scale_max,
            "learnable": bool(self.residual_scale.requires_grad),
        }

    def extra_repr(self):
        state = self.get_scale_state()
        return "scale={}, max={}, learnable={}".format(
            state["effective"], state["max"], state["learnable"]
        )


def build_feature_enhancer(cfg, channels):
    model_cfg = _cfg_get(cfg, "MODEL")
    enhancer_cfg = _cfg_get(model_cfg, "FEATURE_ENHANCER")
    name = str(_cfg_get(enhancer_cfg, "NAME", "none")).lower()

    if name in ("", "none", "identity"):
        return IdentityEnhancer()

    if name not in ("grn", "grn_safe"):
        raise ValueError("Unsupported feature enhancer: {}".format(name))

    cfg_channels = int(_cfg_get(enhancer_cfg, "CHANNELS", channels))
    if cfg_channels != int(channels):
        raise ValueError(
            "FEATURE_ENHANCER.CHANNELS={} does not match backbone channels={}.".format(
                cfg_channels, channels
            )
        )

    grn_cfg = _cfg_get(enhancer_cfg, "GRN")
    grn = GRNSafeEnhancer(
        channels=channels,
        use_bias=_cfg_get(grn_cfg, "USE_BIAS", True),
        eps=_cfg_get(grn_cfg, "EPS", 1e-6),
        detach_response=_cfg_get(grn_cfg, "DETACH_RESPONSE", False),
    )
    return StableResidualAdapter(
        grn,
        scale_init=_cfg_get(enhancer_cfg, "RES_SCALE_INIT", 0.0),
        learnable=_cfg_get(enhancer_cfg, "RES_SCALE_LEARNABLE", True),
        scale_max=_cfg_get(enhancer_cfg, "RES_SCALE_MAX", None),
    )
