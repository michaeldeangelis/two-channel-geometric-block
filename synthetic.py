"""Synthetic task: query-dependent accumulated payload.

Each graph is one center residue plus neighbors. The label is the cutoff-weighted
sum of payloads whose key lies in the center's half-space. Payloads are signed,
so a neighbor count does not determine the label. The match bit depends on the
pair (query, key), and the value network does not see the query.

A uniform mean of per-neighbor features times an ungated count recovers sums of
functions of the neighbor alone. It does not recover this label unless the
softmax mean already contains it.
"""

from __future__ import annotations

import math

import torch

from block import RADIUS, smooth_cutoff

H_DIM = 6  # q_x, q_y, k_x, k_y, payload, ones


def _unit(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, d, generator=gen)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def sample_batch(
    batch: int,
    gen: torch.Generator,
    n_in: int = 24,
    n_decoy: int = 4,
) -> dict[str, torch.Tensor]:
    """Sample a batch. Residue 0 is the center. Label is y.

    Neighborhood size is fixed. Match count still varies with the query, and
    payloads are signed, so an ungated count does not determine the label.
    """
    n_ctx = n_in + n_decoy
    n = 1 + n_ctx
    q = _unit(batch, 2, gen)
    keys = torch.randn(batch, n_ctx, 2, generator=gen)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    payloads = torch.randn(batch, n_ctx, generator=gen)

    r_in = 0.05 + 0.90 * torch.rand(batch, n_in, generator=gen)
    ang_in = (2 * math.pi) * torch.rand(batch, n_in, generator=gen)
    r_out = 1.2 + 1.3 * torch.rand(batch, n_decoy, generator=gen)
    ang_out = (2 * math.pi) * torch.rand(batch, n_decoy, generator=gen)
    r = torch.cat([r_in, r_out], dim=1)
    ang = torch.cat([ang_in, ang_out], dim=1)

    h = torch.zeros(batch, n, H_DIM)
    pos = torch.zeros(batch, n, 2)
    mask = torch.ones(batch, n, dtype=torch.bool)
    h[:, 0, 0:2] = q
    h[:, 0, 5] = 1.0
    h[:, 1:, 2:4] = keys
    h[:, 1:, 4] = payloads
    h[:, 1:, 5] = 1.0
    pos[:, 1:, 0] = r * torch.cos(ang)
    pos[:, 1:, 1] = r * torch.sin(ang)

    c = smooth_cutoff(r, RADIUS)
    match = (keys * q[:, None, :]).sum(dim=-1) > 0
    y = (c * match.to(c.dtype) * payloads).sum(dim=-1)
    return {"h": h, "pos": pos, "mask": mask, "y": y}


def sample_interface_batch(
    batch: int,
    gen: torch.Generator,
    n_same: int = 12,
    n_cross: int = 12,
    n_decoy: int = 4,
) -> dict[str, torch.Tensor]:
    """Label is the product of the same-chain half-space sum and the cross-chain one.

    Chain identity is a pair feature, so an unsplit gated sum can select on it.
    The product asks for both subset sums at once.
    """
    n_near = n_same + n_cross
    n_ctx = n_near + n_decoy
    n = 1 + n_ctx
    q = _unit(batch, 2, gen)
    keys = torch.randn(batch, n_ctx, 2, generator=gen)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    payloads = torch.randn(batch, n_ctx, generator=gen)

    r_near = 0.05 + 0.90 * torch.rand(batch, n_near, generator=gen)
    ang_near = (2 * math.pi) * torch.rand(batch, n_near, generator=gen)
    r_far = 1.2 + 1.3 * torch.rand(batch, n_decoy, generator=gen)
    ang_far = (2 * math.pi) * torch.rand(batch, n_decoy, generator=gen)
    r = torch.cat([r_near, r_far], dim=1)
    ang = torch.cat([ang_near, ang_far], dim=1)

    chain = torch.zeros(batch, n, dtype=torch.long)
    chain[:, 1 : 1 + n_same] = 0
    chain[:, 1 + n_same : 1 + n_near] = 1
    chain[:, 1 + n_near : 1 + n_near + n_decoy // 2] = 0
    chain[:, 1 + n_near + n_decoy // 2 :] = 1

    h = torch.zeros(batch, n, H_DIM)
    pos = torch.zeros(batch, n, 2)
    mask = torch.ones(batch, n, dtype=torch.bool)
    h[:, 0, 0:2] = q
    h[:, 0, 5] = 1.0
    h[:, 1:, 2:4] = keys
    h[:, 1:, 4] = payloads
    h[:, 1:, 5] = 1.0
    pos[:, 1:, 0] = r * torch.cos(ang)
    pos[:, 1:, 1] = r * torch.sin(ang)

    c = smooth_cutoff(r, RADIUS)
    match = (keys * q[:, None, :]).sum(dim=-1) > 0
    nb_chain = chain[:, 1:]
    weight = c * match.to(c.dtype) * payloads
    s_same = (weight * (nb_chain == 0).to(weight.dtype)).sum(dim=-1)
    s_cross = (weight * (nb_chain == 1).to(weight.dtype)).sum(dim=-1)
    y = s_same * s_cross
    return {"h": h, "pos": pos, "mask": mask, "chain": chain, "y": y}
