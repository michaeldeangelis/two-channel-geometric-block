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


def main() -> None:
    test_audit_rejects_a_cross_split_homolog()
    test_zero_coordinates_ignore_geometry()
    test_frequency_baseline_uses_the_training_mode()
    print("scale checks passed")


if __name__ == "__main__":
    main()
