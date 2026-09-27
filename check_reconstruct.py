"""Fixed trained C: rebuild the sum from the mean and log mass, no extra training step.

s = log(1+rho). The rebuilt message is mean * (eps + exp(s) - 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from proteins import ROOT, load_cache
from run_scale import train_one

LOCK = ROOT / "split_magnitude.json"
WIDTH = 80


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    return [by_id[pdb_id] for pdb_id in ids]


def main() -> None:
    lock = json.loads(LOCK.read_text())
    complexes = load_cache()
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    done = train_one("C", "C", WIDTH, True, 0, train, val, test, return_model=True)
    model = done.pop("model")
    model.eval()
    worst = 0.0
    total = 0.0
    count = 0
    from inverse_fold import tensor_batch

    with torch.no_grad():
        for comp in test:
            batch = tensor_batch([comp])
            base = model.logits(batch["pos"], batch["mask"], batch["chain"], batch["resseq"])
            model.block.reconstruct = True
            rebuilt = model.logits(batch["pos"], batch["mask"], batch["chain"], batch["resseq"])
            model.block.reconstruct = False
            err = (base - rebuilt).abs()
            worst = max(worst, float(err.max()))
            total += float(err.sum())
            count += int(err.numel())
    out = {
        "seed": 0,
        "best_step": done["best_step"],
        "max_abs_logit": worst,
        "mean_abs_logit": total / count,
        "n_logits": count,
        "eps": 1e-8,
        "formula": "mean * (eps + exp(log1p(rho)) - 1)",
    }
    path = Path(__file__).resolve().parent / "results" / "reconstruct.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"max abs logit {worst:.3e}  mean {out['mean_abs_logit']:.3e}", flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
