import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class RAFTLargeFlow(nn.Module):
    """
    Frozen torchvision RAFT-large wrapper for optical-flow supervision & debugging.

    Inputs:
        x1, x2: [B, 3, H, W] in [0, 1], RGB, H and W are multiples of 8 (assumed).
    Output:
        flow:   [B, 2, H, W] in pixel units (dx, dy). Optionally normalized by (W, H).

    Notes:
        - No padding/unpadding; we assume /8 divisibility upstream (e.g., Diffusion VAE).
        - Provides flow_to_image() for HSV color-wheel visualization.
    """

    def __init__(self,
                 weights: str | None = 'DEFAULT',
                 num_flow_updates: int = 12,
                 freeze: bool = True):
        super().__init__()

        # Resolve weights enum
        if weights is None or weights == 'DEFAULT':
            weights_enum = Raft_Large_Weights.DEFAULT
        elif isinstance(weights, Raft_Large_Weights):
            weights_enum = weights
        else:
            weights_enum = getattr(Raft_Large_Weights, str(weights))

        self.model = raft_large(weights=weights_enum, progress=False)
        self.num_flow_updates = int(num_flow_updates)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # lightweight device hook
        self.register_buffer("_devhook", torch.empty(0), persistent=False)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
        """
        Args:
            x1, x2 : [B,3,H,W], float in [0,1]
            normalize: if True, divide (dx,dy) by (W,H), making loss resolution-agnostic.

        Returns:
            flow: [B,2,H,W] (pixels or normalized if normalize=True)
        """
        assert x1.shape == x2.shape and x1.dim() == 4 and x1.size(1) == 3, "Expect [B,3,H,W] for x1/x2"
        B, C, H, W = x1.shape
        # Assumption: /8 divisibility ensured upstream; no padding here.
        # (Uncomment the assert if you prefer a hard guard.)
        # assert (H % 8 == 0) and (W % 8 == 0), "RAFT-large expects H,W divisible by 8."

        dev = x1.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)

        # [0,1] -> [-1,1] as expected by torchvision RAFT preprocessing
        x1n = x1.mul(2.0).sub(1.0)
        x2n = x2.mul(2.0).sub(1.0)

        flows_list = self.model(x1n, x2n, num_flow_updates=self.num_flow_updates)  # list of [B,2,H,W]
        flow = flows_list[-1].contiguous()  # take the last (most refined)

        if normalize:
            flow[:, 0].div_(W)  # dx / W
            flow[:, 1].div_(H)  # dy / H
        return flow

    # Convenience alias used by the FloLPIPS evaluator.
    def get_optical_flow(self, x1: torch.Tensor, x2: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
        return self.forward(x1, x2, normalize=normalize)

    # ---------- Debug visualization ----------
    @staticmethod
    @torch.no_grad()
    def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
        """
        Convert optical flow [B,2,H,W] to RGB [B,3,H,W] in [0,1] using HSV color-wheel mapping.
        """
        assert flow.dim() == 4 and flow.size(1) == 2, "flow must be [B,2,H,W]"
        u, v = flow[:, 0], flow[:, 1]  # [B,H,W]

        # magnitude and angle
        rad = torch.sqrt(u * u + v * v + 1e-6)
        ang = torch.atan2(v, u)  # [-pi, pi]

        # map angle to hue in [0,1], magnitude to value
        hue = (ang / (2 * math.pi)) + 0.5
        sat = torch.ones_like(hue)
        # normalize magnitude per-image for visualization
        rad_max = rad.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        val = (rad / rad_max).clamp(0, 1)

        hsv = torch.stack([hue, sat, val], dim=1)  # [B,3,H,W]
        rgb = RAFTLargeFlow._hsv_to_rgb(hsv)
        return rgb.clamp(0, 1)

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        """
        HSV to RGB conversion in [0,1], applied elementwise.
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]  # [B,H,W]
        i = torch.floor(h * 6).to(torch.int32)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i_mod = i % 6
        cond0 = (i_mod == 0)
        cond1 = (i_mod == 1)
        cond2 = (i_mod == 2)
        cond3 = (i_mod == 3)
        cond4 = (i_mod == 4)

        r = torch.where(cond0, v, torch.where(cond1, q, torch.where(cond2, p, torch.where(cond3, p, torch.where(cond4, t, v)))))
        g = torch.where(cond0, t, torch.where(cond1, v, torch.where(cond2, v, torch.where(cond3, q, torch.where(cond4, p, p)))))
        b = torch.where(cond0, p, torch.where(cond1, p, torch.where(cond2, t, torch.where(cond3, v, torch.where(cond4, v, q)))))

        return torch.stack([r, g, b], dim=1)
