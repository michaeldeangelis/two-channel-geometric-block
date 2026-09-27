"""Inverse folding on heterodimers. Amino-acid identity is a label, not a feature.

Comparisons at a matched step budget:
  A vs C   softmax versus gated sum
  C vs D   does attention add anything
  C vs S   does a chain split help the gated sum
  D vs E   does that help depend on attention

D is also trained for twice as many steps. That curve is diagnostic only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from block import TwoChannelBlock, parameter_count

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
H_NODE = 4
H_DIM = 32
MSG = 32
RADIUS_A = 10.0
INTERFACE_A = 8.0


def _std(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1))


class InverseFolder(torch.nn.Module):
    def __init__(self, variant: str, hidden: int, geometry: bool = True):
        super().__init__()
        self.geometry = geometry
        self.lift = torch.nn.Linear(H_NODE, H_DIM)
        self.block = TwoChannelBlock(H_DIM, MSG, MSG, hidden, variant, radius=RADIUS_A)
        self.aa = torch.nn.Linear(H_DIM, len(AA))

    def logits(self, pos, mask, chain, resseq) -> torch.Tensor:
        if not self.geometry:
            pos = torch.zeros_like(pos)
        ang = (2 * math.pi) * resseq / 32.0
        node = torch.stack(
            [
                torch.ones_like(resseq),
                torch.sin(ang),
                torch.cos(ang),
                chain.to(resseq.dtype),
            ],
            dim=-1,
        )
        h = self.lift(node) * mask.unsqueeze(-1).to(node.dtype)
        h_out, _ = self.block.encode(h, pos, mask, chain=chain, resseq=resseq)
        return self.aa(h_out)


def tensor_batch(complexes: list[dict]) -> dict[str, torch.Tensor]:
    lengths = [sum(len(c["seq"]) for c in comp["chains"]) for comp in complexes]
    n = max(lengths)
    b = len(complexes)
    pos = torch.zeros(b, n, 3)
    mask = torch.zeros(b, n, dtype=torch.bool)
    chain = torch.zeros(b, n, dtype=torch.long)
    resseq = torch.zeros(b, n)
    target = torch.zeros(b, n, dtype=torch.long)
    interface = torch.zeros(b, n, dtype=torch.bool)
    for i, comp in enumerate(complexes):
        cursor = 0
        cas: list[list[tuple[float, float, float]]] = []
        spans: list[tuple[int, int]] = []
        for ci, part in enumerate(comp["chains"]):
            m = len(part["seq"])
            spans.append((cursor, cursor + m))
            cas.append(part["xyz"])
            for k in range(m):
                pos[i, cursor + k] = torch.tensor(part["xyz"][k])
                resseq[i, cursor + k] = part["resseq"][k]
                chain[i, cursor + k] = ci
                target[i, cursor + k] = AA_INDEX[part["seq"][k]]
            mask[i, cursor : cursor + m] = True
            cursor += m
        for ci, (start, stop) in enumerate(spans):
            other = cas[1 - ci]
            for k in range(start, stop):
                x, y, z = pos[i, k].tolist()
                near = any(
                    (x - u) ** 2 + (y - v) ** 2 + (z - w) ** 2 <= INTERFACE_A ** 2
                    for u, v, w in other
                )
                interface[i, k] = near
    return {
        "pos": pos,
        "mask": mask,
        "chain": chain,
        "resseq": resseq,
        "target": target,
        "interface": interface,
    }


def loss_and_scores(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    mask = batch["mask"]
    target = batch["target"]
    logp = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    pred = logits.argmax(dim=-1)
    correct = (pred == target) & mask

    def rate(sel: torch.Tensor) -> float:
        denom = int(sel.sum())
        if denom == 0:
            return float("nan")
        return float((correct & sel).sum() / denom)

    def mean_nll(sel: torch.Tensor) -> float:
        denom = int(sel.sum())
        if denom == 0:
            return float("nan")
        return float((logp.detach() * sel).sum() / denom)

    interface = mask & batch["interface"]
    n = int(mask.sum())
    n_interface = int(interface.sum())
    return {
        "nll": mean_nll(mask),
        "recovery": rate(mask),
        "interface": rate(interface),
        "noninterface": rate(mask & ~batch["interface"]),
        "interface_nll": mean_nll(interface),
        "noninterface_nll": mean_nll(mask & ~batch["interface"]),
        "n": n,
        "n_correct": int((correct & mask).sum()),
        "n_interface": n_interface,
        "n_interface_correct": int((correct & interface).sum()),
        "nll_sum": float((logp.detach() * mask).sum()),
        "interface_nll_sum": float((logp.detach() * interface).sum()) if n_interface else 0.0,
    }


def train_background(complexes: list[dict]) -> dict[str, float]:
    """Training-set amino-acid frequencies. No geometry."""
    counts = torch.zeros(len(AA))
    for comp in complexes:
        for part in comp["chains"]:
            for aa in part["seq"]:
                counts[AA_INDEX[aa]] += 1
    freq = counts / counts.sum().clamp_min(1)
    mode = int(freq.argmax())
    return {"freq": freq, "mode": mode, "mode_aa": AA[mode]}


def frequency_scores(batch: dict[str, torch.Tensor], background: dict) -> dict[str, float]:
    freq = background["freq"].to(batch["target"].device)
    logits = freq.clamp_min(1e-8).log().view(1, 1, -1).expand(
        batch["target"].shape[0], batch["target"].shape[1], -1
    )
    return loss_and_scores(logits, batch)


def prediction_mix(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Argmax rates and mean softmax, over masked residues only."""
    mask = batch["mask"]
    pred = logits.argmax(dim=-1)
    chosen = pred[mask]
    counts = torch.bincount(chosen, minlength=len(AA)).to(torch.float)
    rates = counts / counts.sum().clamp_min(1)
    probs = torch.softmax(logits, dim=-1)
    mean_p = (probs * mask.unsqueeze(-1)).sum(dim=(0, 1)) / mask.sum().clamp_min(1)
    return {
        "argmax_freq": [float(x) for x in rates],
        "mean_softmax": [float(x) for x in mean_p],
    }


def hidden_for(variants: tuple[str, ...], base: int = 64) -> dict[str, int]:
    target = max(parameter_count(InverseFolder(v, base)) for v in variants)
    widths = {}
    for variant in variants:
        hidden = base
        while parameter_count(InverseFolder(variant, hidden)) < int(0.90 * target) and hidden < 256:
            hidden += 8
        widths[variant] = hidden
    return widths
