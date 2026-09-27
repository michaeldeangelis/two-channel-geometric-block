"""Amino-acid labels must not change the logits. Geometry may."""

from __future__ import annotations

import torch

from inverse_fold import InverseFolder, tensor_batch


def test_labels_do_not_enter_the_features() -> None:
    torch.manual_seed(0)
    comp = {
        "chains": [
            {"seq": "ACDEFGHIKL", "xyz": [(float(i), 0.0, 0.0) for i in range(10)], "resseq": [float(i) for i in range(10)]},
            {"seq": "LMNPQRSTVW", "xyz": [(float(i), 6.0, 0.0) for i in range(10)], "resseq": [float(i) for i in range(10)]},
        ]
    }
    other = {
        "chains": [
            {"seq": "VVVVVVVVVV", "xyz": [(float(i), 0.0, 0.0) for i in range(10)], "resseq": [float(i) for i in range(10)]},
            {"seq": "AAAAAAAAAA", "xyz": [(float(i), 6.0, 0.0) for i in range(10)], "resseq": [float(i) for i in range(10)]},
        ]
    }
    model = InverseFolder("S", 32)
    model.eval()
    a = tensor_batch([comp])
    b = tensor_batch([other])
    with torch.no_grad():
        la = model.logits(a["pos"], a["mask"], a["chain"], a["resseq"])
        lb = model.logits(b["pos"], b["mask"], b["chain"], b["resseq"])
    if not torch.allclose(la, lb):
        raise AssertionError("amino-acid identity changed the logits")


def main() -> None:
    test_labels_do_not_enter_the_features()
    print("inverse checks passed")


if __name__ == "__main__":
    main()
