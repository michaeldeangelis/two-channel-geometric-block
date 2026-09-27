"""Behavior checks for the block. These do not show biological usefulness."""

from __future__ import annotations

import math

import torch

from block import TwoChannelBlock, smooth_cutoff


def _identical_case(n_neighbors: int, h_dim: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = n_neighbors + 1
    h = torch.ones(1, n, h_dim)
    e = torch.zeros(1, n, n, 3)
    c = torch.ones(1, n, n)
    mask = torch.ones(1, n, dtype=torch.bool)
    pos = torch.zeros(1, n, 2)
    return h, pos, mask, e, c


def test_softmax_collapses_identical_values() -> None:
    torch.manual_seed(0)
    block = TwoChannelBlock(h_dim=6, hidden=32, variant="D")
    block.eval()
    out = []
    sums = []
    for n_nb in (2, 20):
        h, pos, mask, e, c = _identical_case(n_nb)
        _y, msg = block(h, pos, mask, e_override=e, c_override=c)
        out.append(msg["m_select"][0, 0].detach())
        sums.append(msg["m_sum"][0, 0].detach())
    if not torch.allclose(out[0], out[1], atol=1e-5):
        raise AssertionError(f"softmax separated identical neighborhoods:\n{out[0]}\n{out[1]}")
    ratio = sums[1].norm() / sums[0].norm().clamp_min(1e-8)
    # 21 nodes vs 3 nodes, all pairs open, gates equal → sum scales with node count.
    expected = 21 / 3
    if abs(ratio.item() - expected) > 0.05 * expected:
        raise AssertionError(f"gated sum did not scale with count: ratio {ratio.item():.3f} vs {expected:.3f}")


def test_distant_node_leaves_local_channels() -> None:
    torch.manual_seed(1)
    block = TwoChannelBlock(h_dim=6, hidden=32, variant="D")
    block.eval()
    h = torch.randn(1, 6, 6)
    pos = torch.randn(1, 6, 2) * 0.3
    pos[0, 0] = 0
    mask = torch.ones(1, 6, dtype=torch.bool)
    _y0, m0 = block(h, pos, mask)

    h2 = torch.zeros(1, 7, 6)
    pos2 = torch.zeros(1, 7, 2)
    mask2 = torch.zeros(1, 7, dtype=torch.bool)
    h2[0, :6] = h[0]
    pos2[0, :6] = pos[0]
    mask2[0, :6] = True
    h2[0, 6] = torch.tensor([4.0, -3.0, 2.0, 2.0, 9.0, 1.0])
    pos2[0, 6] = torch.tensor([8.0, -6.0])
    mask2[0, 6] = True
    _y1, m1 = block(h2, pos2, mask2)
    for key in ("m_select", "m_sum", "rho"):
        a = m0[key][0, 0]
        b = m1[key][0, 0]
        if not torch.allclose(a, b, atol=1e-5):
            raise AssertionError(f"{key} changed when a distant node was added: {(a - b).abs().max().item()}")


def test_rotation_and_translation_leave_scalar() -> None:
    torch.manual_seed(2)
    block = TwoChannelBlock(h_dim=6, hidden=32, variant="D")
    block.eval()
    h = torch.randn(1, 8, 6)
    pos = torch.randn(1, 8, 2) * 0.4
    mask = torch.ones(1, 8, dtype=torch.bool)
    y0, _ = block(h, pos, mask)
    theta = 0.7
    rot = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    pos_r = pos @ rot.T + torch.tensor([3.0, -2.0])
    y1, _ = block(h, pos_r, mask)
    if not torch.allclose(y0, y1, atol=1e-5):
        raise AssertionError(f"scalar changed under rigid motion: {y0.item()} vs {y1.item()}")


def test_density_feature_is_wired() -> None:
    torch.manual_seed(3)
    block = TwoChannelBlock(h_dim=6, msg_dim=4, hidden=8, variant="D")
    # Zero every fuse weight, then put 1 on the density slot so y tracks log1p(rho)
    # only through the head's read of that residual channel.
    with torch.no_grad():
        for p in block.fuse.parameters():
            p.zero_()
        # fuse output = W2 @ gelu(W1 x + b1) + b2. Set W1 so the first hidden unit
        # copies the density feature, gelu'(0) path: use a large positive pre-activation
        # by writing the density into a skip we control through the head instead.
        block.head.weight.zero_()
        block.head.bias.zero_()
    h = torch.zeros(1, 4, 6)
    h[..., 5] = 1
    pos = torch.zeros(1, 4, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    e = torch.zeros(1, 4, 4, 3)
    c = torch.ones(1, 4, 4)
    _y, msg = block(h, pos, mask, e_override=e, c_override=c)
    if msg["rho"][0, 0].item() <= 1:
        raise AssertionError("rho did not accumulate across the fully connected neighborhood")
    # Density slot is the last fuse input. A one-hot fuse row must change y.
    with torch.no_grad():
        block.fuse.net[0].weight.zero_()
        block.fuse.net[0].bias.zero_()
        block.fuse.net[0].bias[0] = 2.0  # gelu(2) > 0 even with zero weights
        block.fuse.net[2].weight.zero_()
        block.fuse.net[2].bias.zero_()
        block.fuse.net[2].weight[0, 0] = 1.0
        block.head.weight.zero_()
        block.head.weight[0, 0] = 1.0
    y_base, msg = block(h, pos, mask, e_override=e, c_override=c)
    # Now route the density feature into hidden unit 0 instead of the constant bias.
    with torch.no_grad():
        block.fuse.net[0].bias.zero_()
        density_index = 6 + 4 + 4  # h_dim + msg + msg
        block.fuse.net[0].weight.zero_()
        block.fuse.net[0].weight[0, density_index] = 1.0
    y_rho, _ = block(h, pos, mask, e_override=e, c_override=c)
    if torch.allclose(y_base, y_rho, atol=1e-6):
        raise AssertionError("log(1+rho) is not connected to the scalar head")
    if y_rho.item() <= 0:
        raise AssertionError(f"expected positive readout from log1p(rho), got {y_rho.item()}")


def test_cutoff_is_zero_outside_radius() -> None:
    r = torch.tensor([0.0, 0.5, 1.0, 1.01, 3.0])
    c = smooth_cutoff(r)
    if c[0].item() < 0.99 or c[-1].item() != 0.0 or c[-2].item() != 0.0:
        raise AssertionError(f"cutoff shape wrong: {c.tolist()}")


def main() -> None:
    test_cutoff_is_zero_outside_radius()
    test_softmax_collapses_identical_values()
    test_distant_node_leaves_local_channels()
    test_rotation_and_translation_leave_scalar()
    test_density_feature_is_wired()
    print("behavior checks passed")


if __name__ == "__main__":
    main()
