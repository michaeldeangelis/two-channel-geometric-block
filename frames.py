"""Backbone-frame edges for a stronger inverse folder.

The neighbor graph is still the 10 Å CA window. What changes is the edge
vector inside that window: a distance RBF, the neighbor CA in the local
N-CA-C frame, and the relative rotation of the two frames. Sequence
separation and chain identity stay in the last two channels.

structure=False keeps the same graph and those last two channels, and
zeros the geometric channels. It does not move the coordinates, so it
does not turn the window into a complete graph.
"""

from __future__ import annotations

import torch

from block import smooth_cutoff
from proteins import PDB_DIR

RADIUS_A = 10.0
RBF_CENTERS = 16
RBF_MAX = 20.0
GEOM_DIM = RBF_CENTERS + 3 + 9
E_DIM = GEOM_DIM + 2


def rigid_frames(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Columns are the N-CA-C basis. Shape (B, N, 3, 3)."""
    e1 = c - ca
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = n - ca
    e2 = v - e1 * (e1 * v).sum(dim=-1, keepdim=True)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def edge_features(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor,
    chain: torch.Tensor,
    resseq: torch.Tensor,
    radius: float = RADIUS_A,
) -> tuple[torch.Tensor, torch.Tensor]:
    """e is (B, N, N, 30). c is the smooth 10 Å window, masked."""
    R = rigid_frames(n, ca, c)
    delta = ca[:, None, :, :] - ca[:, :, None, :]
    dist = delta.norm(dim=-1)
    local = torch.einsum("biak,bija->bijk", R, delta) / radius
    rel = torch.einsum("biak,bjam->bijkm", R, R).flatten(-2)
    centers = torch.linspace(0, RBF_MAX, RBF_CENTERS, device=ca.device, dtype=ca.dtype)
    sigma = RBF_MAX / (RBF_CENTERS - 1)
    rbf = torch.exp(-((dist.unsqueeze(-1) - centers) / sigma) ** 2)
    same = (chain[:, :, None] == chain[:, None, :]).to(ca.dtype)
    sep = (resseq[:, :, None] - resseq[:, None, :]).abs() / 32.0
    sep = torch.where(same > 0.5, sep, torch.ones_like(sep))
    e = torch.cat([rbf, local, rel, sep.unsqueeze(-1), same.unsqueeze(-1)], dim=-1)
    window = smooth_cutoff(dist, radius)
    valid = mask[:, :, None] & mask[:, None, :]
    return e, window * valid.to(ca.dtype)


def zero_geometry(e: torch.Tensor) -> torch.Tensor:
    out = e.clone()
    out[..., :GEOM_DIM] = 0
    return out


def attach_backbone(comp: dict) -> bool:
    """Add N and C coordinates aligned to the cached CA residues. False if any atom is missing."""
    path = PDB_DIR / f"{comp['id']}.pdb"
    if not path.exists():
        return False
    atoms: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        name = line[12:16].strip()
        if name not in {"N", "CA", "C"}:
            continue
        chain = line[21]
        resi = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        atoms[(chain, resi, name)] = xyz
    for part in comp["chains"]:
        ns = []
        cs = []
        for resi in part["resseq"]:
            key_n = (part["chain"], int(resi), "N")
            key_c = (part["chain"], int(resi), "C")
            if key_n not in atoms or key_c not in atoms:
                return False
            ns.append(atoms[key_n])
            cs.append(atoms[key_c])
        part["n"] = ns
        part["c"] = cs
    return True
