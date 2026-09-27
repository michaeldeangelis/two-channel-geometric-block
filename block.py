"""Two-channel geometric block. Variants A–D plus diagnostic R.

A  geometric softmax
B  softmax plus ungated neighbor-count features
C  gated sum only, plus log(1 + rho)
D  softmax plus gated sum, plus log(1 + rho)
R  softmax plus gated log(1 + rho), no vector sum (diagnostic)
E  D, with same-chain and cross-chain channels kept separate
S  C, with those channels kept separate. No attention.

R is not part of the pass rule. It asks whether a gated count alone
is doing the work of the vector sum.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

RADIUS = 1.0


def smooth_cutoff(r: torch.Tensor, radius: float = RADIUS) -> torch.Tensor:
    """Smooth window. Equals 1 at r=0 and 0 for r >= radius."""
    u = (r / radius).clamp(0, 1)
    c = 0.5 * (1.0 + torch.cos(math.pi * u))
    return torch.where(r <= radius, c, torch.zeros_like(r))


def pair_geometry(
    pos: torch.Tensor,
    mask: torch.Tensor,
    chain: torch.Tensor | None = None,
    radius: float = RADIUS,
    resseq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation-invariant pair features and the smooth cutoff.

    pos: (B, N, 3) or (B, N, 2). mask: (B, N) bool. chain, resseq: (B, N).
    Returns e (B, N, N, 3) = [distance/radius, sequence separation, same-chain]
    and c (B, N, N). With the default radius of 1, distance is unchanged.
    """
    b, n, _ = pos.shape
    delta = pos[:, :, None, :] - pos[:, None, :, :]
    dist = torch.linalg.norm(delta, dim=-1)
    if chain is None:
        same = torch.ones(b, n, n, device=pos.device, dtype=pos.dtype)
    else:
        same = (chain[:, :, None] == chain[:, None, :]).to(pos.dtype)
    if resseq is None:
        idx = torch.arange(n, device=pos.device, dtype=pos.dtype)
        sep = (idx[None, :] - idx[:, None]).abs() / 32.0
        sep = sep.view(1, n, n).expand(b, n, n)
    else:
        sep = (resseq[:, :, None] - resseq[:, None, :]).abs() / 32.0
        # A cross-chain pair is not a sequence neighbor.
        sep = torch.where(same > 0.5, sep, torch.ones_like(sep))
    e = torch.stack([dist / radius, sep, same], dim=-1)
    c = smooth_cutoff(dist, radius)
    valid = mask[:, :, None] & mask[:, None, :]
    c = c * valid.to(pos.dtype)
    return e, c


class MLP(nn.Module):
    def __init__(self, din: int, hidden: int, dout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.GELU(),
            nn.Linear(hidden, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoChannelBlock(nn.Module):
    """One block plus a scalar head on residue 0.

    h is the residue state. Geometry is computed from positions.
    """

    def __init__(
        self,
        h_dim: int,
        msg_dim: int = 16,
        attn_dim: int = 16,
        hidden: int = 64,
        variant: str = "D",
        e_dim: int = 3,
        radius: float = RADIUS,
    ):
        super().__init__()
        if variant not in {"A", "B", "C", "D", "R", "E", "S", "N", "Q"}:
            raise ValueError(variant)
        self.variant = variant
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        self.attn_dim = attn_dim
        self.radius = radius
        # N and Q divide the gated sum by gate mass. Q also keeps log(1+rho).
        self.normalize = variant in {"N", "Q"}
        self.split = variant in {"E", "S"}
        self.use_select = variant in {"A", "B", "D", "R", "E"}
        self.use_sum = variant in {"C", "D", "E", "S", "N", "Q"}
        self.density = {
            "A": "none",
            "B": "count",
            "C": "gated",
            "D": "gated",
            "R": "gated",
            "E": "gated",
            "S": "gated",
            "N": "none",
            "Q": "gated",
        }[variant]
        self.need_gate = self.use_sum or self.density == "gated"

        if self.use_select:
            self.q = nn.Linear(h_dim, attn_dim, bias=False)
            self.k = nn.Linear(h_dim, attn_dim, bias=False)
            self.bias = nn.Linear(e_dim, 1)
            self.value = MLP(h_dim + e_dim, hidden, msg_dim)
        if self.need_gate:
            self.gate = MLP(h_dim + h_dim + e_dim, hidden, 1)
        if self.use_sum:
            self.phi = MLP(h_dim + e_dim, hidden, msg_dim)
        if variant == "S":
            # Split sums only. Empty attention slots would pad the budget.
            fuse_in = h_dim + 2 * (msg_dim + 1)
        elif self.split:
            fuse_in = h_dim + 2 * (msg_dim + msg_dim + 1)
        else:
            fuse_in = h_dim + msg_dim + msg_dim + 1
        self.fuse = MLP(fuse_in, hidden, h_dim)
        self.head = nn.Linear(h_dim, 1)

    def messages(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_override: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
        chain: torch.Tensor | None = None,
        resseq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return pre-fusion channels at every residue. Shape (B, N, ...)."""
        if e_override is None:
            e, c = pair_geometry(pos, mask, chain, self.radius, resseq)
        else:
            e, c = e_override, c_override
        return self._channels(h, e, c)

    def _channels(self, h: torch.Tensor, e: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, _ = h.shape
        h_i = h[:, :, None, :].expand(b, n, n, self.h_dim)
        h_j = h[:, None, :, :].expand(b, n, n, self.h_dim)
        count = c.sum(dim=-1)

        if self.use_select:
            scores = torch.matmul(self.q(h), self.k(h).transpose(-1, -2)) / math.sqrt(self.attn_dim)
            scores = scores + self.bias(e).squeeze(-1)
            log_c = torch.where(c > 0, c.clamp_min(1e-8).log(), torch.full_like(c, -1e9))
            weights = torch.softmax(scores + log_c, dim=-1)
            weights = torch.where(c > 0, weights, torch.zeros_like(weights))
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
            v = self.value(torch.cat([h_j, e], dim=-1))
            m_select = torch.einsum("bij,bijm->bim", weights, v)
        else:
            weights = torch.zeros(b, n, n, device=h.device, dtype=h.dtype)
            m_select = torch.zeros(b, n, self.msg_dim, device=h.device, dtype=h.dtype)

        if self.need_gate:
            g = torch.sigmoid(self.gate(torch.cat([h_i, h_j, e], dim=-1))).squeeze(-1)
        else:
            g = torch.zeros(b, n, n, device=h.device, dtype=h.dtype)
        if self.use_sum:
            phi = self.phi(torch.cat([h_j, e], dim=-1))
            m_sum = torch.einsum("bij,bijm->bim", c * g, phi)
        else:
            m_sum = torch.zeros(b, n, self.msg_dim, device=h.device, dtype=h.dtype)
        rho = (c * g).sum(dim=-1) if self.need_gate else torch.zeros(b, n, device=h.device, dtype=h.dtype)
        if self.normalize:
            # m_mean = sum_j g v / (eps + sum_j g). eps is 1e-8.
            m_sum = m_sum / (1e-8 + rho).unsqueeze(-1)
        return {
            "m_select": m_select,
            "m_sum": m_sum,
            "rho": rho,
            "count": count,
            "weights": weights,
            "c": c,
        }

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_override: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
        chain: torch.Tensor | None = None,
        resseq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_out, msg = self.encode(h, pos, mask, e_override, c_override, chain, resseq)
        y = self.head(h_out[:, 0, :]).squeeze(-1)
        return y, msg

    def encode(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        e_override: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
        chain: torch.Tensor | None = None,
        resseq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.split:
            if e_override is None:
                e, c = pair_geometry(pos, mask, chain, self.radius, resseq)
            else:
                e, c = e_override, c_override
            same = e[..., 2]
            msg = self._channels(h, e, c * same)
            cross = self._channels(h, e, c * (1.0 - same))
            if self.use_select:
                fuse_in = torch.cat(
                    [
                        h,
                        msg["m_select"],
                        msg["m_sum"],
                        torch.log1p(msg["rho"]).unsqueeze(-1),
                        cross["m_select"],
                        cross["m_sum"],
                        torch.log1p(cross["rho"]).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                fuse_in = torch.cat(
                    [
                        h,
                        msg["m_sum"],
                        torch.log1p(msg["rho"]).unsqueeze(-1),
                        cross["m_sum"],
                        torch.log1p(cross["rho"]).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            h_out = h + self.fuse(fuse_in)
            msg["rho_cross"] = cross["rho"]
            msg["m_sum_cross"] = cross["m_sum"]
            return h_out, msg

        msg = self.messages(h, pos, mask, e_override, c_override, chain, resseq)
        if self.density == "none":
            density = torch.zeros_like(msg["rho"])
        elif self.density == "count":
            density = msg["count"]
        else:
            density = msg["rho"]
        m_select = msg["m_select"] if self.use_select else torch.zeros_like(msg["m_select"])
        m_sum = msg["m_sum"] if self.use_sum else torch.zeros_like(msg["m_sum"])
        fuse_in = torch.cat([h, m_select, m_sum, torch.log1p(density).unsqueeze(-1)], dim=-1)
        h_out = h + self.fuse(fuse_in)
        return h_out, msg


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
