import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TrackGuidedFusionNet(nn.Module):
    """Small post-processing network for F_refined = F0 + gate * DeltaF."""

    def __init__(self, flow_channels=2, prior_channels=5, hidden_dim=48):
        super().__init__()
        in_channels = flow_channels + prior_channels
        self.enc1 = ConvBlock(in_channels, hidden_dim)
        self.enc2 = ConvBlock(hidden_dim, hidden_dim * 2)
        self.enc3 = ConvBlock(hidden_dim * 2, hidden_dim * 4)

        self.dec2 = ConvBlock(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2)
        self.dec1 = ConvBlock(hidden_dim * 2 + hidden_dim, hidden_dim)

        self.delta_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, flow, track_prior):
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"Expected flow with shape (B, 2, H, W), got {flow.shape}")
        if track_prior.ndim != 4:
            raise ValueError(f"Expected track_prior with shape (B, C, H, W), got {track_prior.shape}")
        if flow.shape[-2:] != track_prior.shape[-2:]:
            raise ValueError(f"flow and track_prior spatial sizes differ: {flow.shape[-2:]} vs {track_prior.shape[-2:]}")

        x = torch.cat([flow, track_prior], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))

        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        delta = self.delta_head(d1)
        gate = torch.sigmoid(self.gate_head(d1))
        refined = flow + gate * delta

        return {
            "refined_flow": refined,
            "delta_flow": delta,
            "gate": gate,
        }
