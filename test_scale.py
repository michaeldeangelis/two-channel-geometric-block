"""Split audit, frequency baseline, and the coordinate ablation."""

from __future__ import annotations

import torch

from inverse_fold import InverseFolder, frequency_scores, tensor_batch, train_background
from proteins import audit_partitions, cluster, split_three


def _comp(pdb_id: str, seq_a: str, seq_b: str, gap: float) -> dict:
    def chain(seq: str, y: float) -> dict:
        return {
            "seq": seq,
            "xyz": [(float(i), y, 0.0) for i in range(len(seq))],
            "resseq": [float(i) for i in range(len(seq))],
        }
    return {"id": pdb_id, "chains": [chain(seq_a, 0.0), chain(seq_b, gap)]}


def test_audit_rejects_a_cross_split_homolog() -> None:
    same = "ACDEFGHIKLMNPQRSTVWY" * 3
    other = "VVVVVVVVVVVVVVVVVVVV" * 3
    complexes = [
        _comp("7OWD", same, other, 6),
        _comp("NEW1", same, other, 6),
        _comp("NEW2", "WWWWWWWWWWWWWWWWWWWW", "YYYYYYYYYYYYYYYYYYYY", 6),
        _comp("NEW3", "DDDDDDDDDDDDDDDDDDDD", "EEEEEEEEEEEEEEEEEEEE", 6),
        _comp("NEW4", "GGGGGGGGGGGGGGGGGGGG", "HHHHHHHHHHHHHHHHHHHH", 6),
    ]
    # Plant NEW1 in test even though it matches 7OWD.
    split = {"train": [0], "val": [2], "test": [1]}
    audit = audit_partitions(complexes, split)
    if audit["passed"]:
        raise AssertionError("a copied chain crossed the split undetected")
    grouped = split_three(complexes, cluster(complexes))
    held = audit_partitions(complexes, grouped)
    if not held["passed"]:
        raise AssertionError(held["violations"])
    if complexes[grouped["test"][0]]["id"] == "7OWD":
        raise AssertionError("stage 3 test id landed in test")


def test_zero_coordinates_rebuild_the_graph() -> None:
    """A 40 Å pair is outside a 10 Å window until coordinates are zeroed."""
    from block import pair_geometry

    pos = torch.tensor([[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    _, far = pair_geometry(pos, mask, radius=10.0)
    _, near = pair_geometry(torch.zeros_like(pos), mask, radius=10.0)
    if float(far[0, 0, 1]) != 0.0:
        raise AssertionError("a 40 Å pair was inside the 10 Å window")
    if float(near[0, 0, 1]) < 0.99:
        raise AssertionError("zero coordinates did not put the pair inside the window")


def test_zero_coordinates_ignore_geometry() -> None:
    torch.manual_seed(0)
    seq = "ACDEFGHIKL"
    a = _comp("A", seq, "LMNPQRSTVW", 6)
    b = _comp("B", seq, "LMNPQRSTVW", 40)
    model = InverseFolder("C", 32, geometry=False)
    model.eval()
    with torch.no_grad():
        la = model.logits(*_xy(a))
        lb = model.logits(*_xy(b))
    if not torch.allclose(la, lb):
        raise AssertionError("coordinates changed logits after the geometry ablation")
    geometric = InverseFolder("C", 32, geometry=True)
    geometric.load_state_dict(model.state_dict())
    geometric.eval()
    with torch.no_grad():
        ga = geometric.logits(*_xy(a))
        gb = geometric.logits(*_xy(b))
    if torch.allclose(ga, gb):
        raise AssertionError("moving a chain 40 Å did not change the geometric model")


def test_frequency_baseline_uses_the_training_mode() -> None:
    complexes = [
        _comp("T1", "LLLLLLLLLL", "AAAAAAAAAA", 0),
        _comp("T2", "LLLLLLLLLL", "CCCCCCCCCC", 0),
    ]
    background = train_background(complexes)
    if background["mode_aa"] != "L":
        raise AssertionError(background["mode_aa"])
    batch = tensor_batch([complexes[0]])
    scores = frequency_scores(batch, background)
    # 10/20 residues are leucine.
    if abs(scores["recovery"] - 0.5) > 1e-6:
        raise AssertionError(scores["recovery"])


def _xy(comp: dict):
    batch = tensor_batch([comp])
    return batch["pos"], batch["mask"], batch["chain"], batch["resseq"]


def test_frame_edges_are_rigid_motions() -> None:
    from frames import GEOM_DIM, edge_features, zero_geometry

    torch.manual_seed(0)
    n = torch.randn(1, 5, 3)
    ca = n + torch.tensor([1.4, 0.2, 0.0])
    c = ca + torch.tensor([0.1, 1.5, 0.0])
    mask = torch.ones(1, 5, dtype=torch.bool)
    chain = torch.zeros(1, 5, dtype=torch.long)
    chain[:, 3:] = 1
    resseq = torch.arange(5).float().view(1, -1)
    e0, c0 = edge_features(n, ca, c, mask, chain, resseq)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([4.0, -2.0, 0.5])
    e1, c1 = edge_features(n @ q + shift, ca @ q + shift, c @ q + shift, mask, chain, resseq)
    if not torch.allclose(e0, e1, atol=1e-4) or not torch.allclose(c0, c1, atol=1e-5):
        raise AssertionError(float((e0 - e1).abs().max()))
    plain = zero_geometry(e0)
    if float(plain[..., :GEOM_DIM].abs().max()) != 0.0:
        raise AssertionError("geometric channels stayed on")
    if not torch.equal(plain[..., GEOM_DIM:], e0[..., GEOM_DIM:]):
        raise AssertionError("sequence channels changed")


def test_log_mass_reconstruction_matches_logits() -> None:
    from inverse_fold import InverseFolder

    torch.manual_seed(1)
    model = InverseFolder("C", 32)
    model.eval()
    pos = torch.randn(2, 9, 3) * 8
    mask = torch.ones(2, 9, dtype=torch.bool)
    chain = torch.zeros(2, 9, dtype=torch.long)
    chain[:, 5:] = 1
    resseq = torch.arange(9).float().view(1, -1).expand(2, -1)
    with torch.no_grad():
        base = model.logits(pos, mask, chain, resseq)
        model.block.reconstruct = True
        rebuilt = model.logits(pos, mask, chain, resseq)
    gap = (base - rebuilt).abs().max()
    if float(gap) > 1e-4:
        raise AssertionError(float(gap))


def test_normalized_mean_reconstructs_the_gated_sum() -> None:
    from block import TwoChannelBlock

    torch.manual_seed(0)
    pos = torch.randn(1, 6, 3)
    mask = torch.ones(1, 6, dtype=torch.bool)
    h = torch.randn(1, 6, 32)
    summed = TwoChannelBlock(32, 32, 32, 16, "C", radius=10.0)
    averaged = TwoChannelBlock(32, 32, 32, 16, "N", radius=10.0)
    with_mass = TwoChannelBlock(32, 32, 32, 16, "Q", radius=10.0)
    averaged.load_state_dict(summed.state_dict())
    with_mass.load_state_dict(summed.state_dict())
    with torch.no_grad():
        got_sum = summed.messages(h, pos, mask)
        got_mean = averaged.messages(h, pos, mask)
        got_mass = with_mass.messages(h, pos, mask)
    recon = got_mean["m_sum"] * (1e-8 + got_mean["rho"]).unsqueeze(-1)
    if not torch.allclose(recon, got_sum["m_sum"], atol=1e-5):
        raise AssertionError("mean times gate mass did not recover the sum")
    if not torch.allclose(got_mean["m_sum"], got_mass["m_sum"], atol=1e-5):
        raise AssertionError("N and Q changed the gated messages")


def test_cluster_interval_is_constant_when_every_cluster_agrees() -> None:
    from run_confirm import cluster_interval

    estimate = cluster_interval([[-0.02], [-0.02, -0.02]])
    if abs(estimate["point"] + 0.02) > 1e-12:
        raise AssertionError(estimate["point"])
    if abs(estimate["lo"] - estimate["hi"]) > 1e-12:
        raise AssertionError(estimate)


def main() -> None:
    test_audit_rejects_a_cross_split_homolog()
    test_zero_coordinates_rebuild_the_graph()
    test_zero_coordinates_ignore_geometry()
    test_frequency_baseline_uses_the_training_mode()
    test_cluster_interval_is_constant_when_every_cluster_agrees()
    test_normalized_mean_reconstructs_the_gated_sum()
    test_log_mass_reconstruction_matches_logits()
    test_frame_edges_are_rigid_motions()
    print("scale checks passed")


if __name__ == "__main__":
    main()
