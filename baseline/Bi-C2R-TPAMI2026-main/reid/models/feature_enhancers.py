import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityEnhancer(nn.Module):
    """No-op feature enhancer for the original baseline."""

    def forward(self, x):
        return x


class FUSEChannelAttention(nn.Module):
    """Channel attention from spatial and frequency-domain statistics.

    The module preserves the input feature shape [B, C, H, W]. Ablation flags
    keep the public interface stable while removing individual signal sources.
    """

    def __init__(
        self,
        channels,
        reduction=16,
        lf_ratio=0.125,
        eps=1e-6,
        use_std=True,
        use_freq=True,
        use_tau=True,
        use_per_channel_weight=True,
    ):
        super(FUSEChannelAttention, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive, got {}".format(channels))

        self.channels = int(channels)
        self.reduction = max(1, int(reduction))
        self.lf_ratio = float(lf_ratio)
        self.eps = float(eps)
        self.use_std = bool(use_std)
        self.use_freq = bool(use_freq)
        self.use_tau = bool(use_tau)
        self.use_per_channel_weight = bool(use_per_channel_weight)

        if self.use_per_channel_weight:
            self.per_channel_weight = nn.Parameter(torch.full((self.channels, 4), 0.25))
            self.per_channel_bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("per_channel_weight", None)
            self.register_parameter("per_channel_bias", None)

        hidden = max(1, self.channels // self.reduction)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.channels, bias=True),
        )

        if self.use_tau:
            self.tau_proj = nn.Linear(1, 1, bias=True)
            nn.init.constant_(self.tau_proj.weight, 1.0)
            nn.init.constant_(self.tau_proj.bias, 0.0)
        else:
            self.tau_proj = None

    def _spatial_stats(self, x):
        mean = x.mean(dim=(2, 3))

        if self.use_std or self.use_tau:
            var = (x - mean[:, :, None, None]).pow(2).mean(dim=(2, 3))
            std_raw = (var + self.eps).sqrt()
        else:
            std_raw = torch.zeros_like(mean)

        std_for_stats = std_raw if self.use_std else torch.zeros_like(mean)
        return mean, std_for_stats, std_raw

    def _freq_stats(self, x):
        batch, channels, height, width = x.shape
        zeros = x.new_zeros(batch, channels)
        if not self.use_freq:
            return zeros, zeros

        low_precision_dtypes = [torch.float16]
        if hasattr(torch, "bfloat16"):
            low_precision_dtypes.append(torch.bfloat16)
        fft_input = x.float() if x.dtype in low_precision_dtypes else x
        if hasattr(torch, "fft") and hasattr(torch.fft, "rfft2"):
            mag = torch.fft.rfft2(fft_input, norm="ortho").abs()
        elif hasattr(torch, "rfft"):
            freq = torch.rfft(fft_input, signal_ndim=2, normalized=True, onesided=True)
            mag = (freq.pow(2).sum(dim=-1) + self.eps).sqrt()
        else:
            raise RuntimeError("FUSEChannelAttention requires torch.fft.rfft2 or torch.rfft")

        mag = mag.to(dtype=x.dtype)
        freq_width = width // 2 + 1
        k = max(1, int(min(height, width) * self.lf_ratio))
        kh = min(height, k)
        kw = min(freq_width, k)

        low = mag[:, :, :kh, :kw]
        lfe = low.mean(dim=(2, 3))

        total_count = height * freq_width
        low_count = kh * kw
        high_count = total_count - low_count
        if high_count > 0:
            high_sum = mag.sum(dim=(2, 3)) - low.sum(dim=(2, 3))
            hfe = (high_sum / float(high_count)).clamp_min(0.0)
        else:
            hfe = zeros

        return lfe, hfe

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D feature map [B, C, H, W], got {}".format(tuple(x.shape)))
        if x.size(1) != self.channels:
            raise ValueError("expected {} channels, got {}".format(self.channels, x.size(1)))

        mean, std, std_for_tau = self._spatial_stats(x)
        lfe, hfe = self._freq_stats(x)
        stats = torch.stack([mean, std, lfe, hfe], dim=-1)

        if self.use_per_channel_weight:
            seed = (stats * self.per_channel_weight.unsqueeze(0)).sum(dim=-1)
            seed = seed + self.per_channel_bias.unsqueeze(0)
        else:
            seed = stats.mean(dim=-1)

        logits = self.mlp(seed)
        if self.use_tau:
            uncertainty = std_for_tau.mean(dim=1, keepdim=True)
            tau = F.softplus(self.tau_proj(uncertainty)) + 1e-3
            logits = logits / tau

        gates = torch.sigmoid(logits)
        return x * gates[:, :, None, None]


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

    configured_channels = int(_cfg_get(feature_cfg, "CHANNELS", channels))
    if configured_channels != int(channels):
        raise ValueError(
            "FEATURE_ENHANCER.CHANNELS ({}) does not match backbone channels ({})".format(
                configured_channels, channels
            )
        )

    if name != "fuse":
        raise ValueError("unsupported feature enhancer: {}".format(name))

    fuse_cfg = _cfg_get(feature_cfg, "FUSE")
    fuse = FUSEChannelAttention(
        channels=channels,
        reduction=int(_cfg_get(fuse_cfg, "REDUCTION", 16)),
        lf_ratio=float(_cfg_get(fuse_cfg, "LF_RATIO", 0.125)),
        eps=float(_cfg_get(fuse_cfg, "EPS", 1e-6)),
        use_std=_cfg_bool(_cfg_get(fuse_cfg, "USE_STD", True)),
        use_freq=_cfg_bool(_cfg_get(fuse_cfg, "USE_FREQ", True)),
        use_tau=_cfg_bool(_cfg_get(fuse_cfg, "USE_TAU", True)),
        use_per_channel_weight=_cfg_bool(_cfg_get(fuse_cfg, "USE_PER_CHANNEL_WEIGHT", True)),
    )
    return StableResidualAdapter(
        fuse,
        res_scale_init=float(_cfg_get(feature_cfg, "RES_SCALE_INIT", 0.0)),
        learnable=_cfg_bool(_cfg_get(feature_cfg, "RES_SCALE_LEARNABLE", True)),
    )


__all__ = [
    "IdentityEnhancer",
    "FUSEChannelAttention",
    "StableResidualAdapter",
    "build_feature_enhancer",
]
