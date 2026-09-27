"""Stage 3. Matched-budget inverse folding on heterodimers.

± in the written result is the sample standard deviation across seeds.
The doubled D run is recorded separately and is not the comparison.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from inverse_fold import InverseFolder, hidden_for, loss_and_scores, tensor_batch
from proteins import ROOT, build_cache, cluster, load_cache, split_clusters

STEPS = 200
DIAGNOSTIC_VARIANTS = {"D"}
SEEDS = (0, 1, 2)
VARIANTS = ("A", "C", "S", "D", "E")
BATCH = 2


def _std(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5


def evaluate(model: InverseFolder, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model.logits(batch["pos"], batch["mask"], batch["chain"], batch["resseq"])
        return loss_and_scores(logits, batch)


def train_one(variant: str, hidden: int, seed: int, train, test_batch, steps: int) -> dict:
    torch.manual_seed(seed)
    model = InverseFolder(variant, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    order = list(range(len(train)))
    for step in range(1, steps + 1):
        if (step - 1) % max(len(order) // BATCH, 1) == 0:
            random.Random(seed + step).shuffle(order)
        pick = [train[order[k % len(order)]] for k in range(step - 1, step - 1 + BATCH)]
        batch = tensor_batch(pick)
        model.train()
        logits = model.logits(batch["pos"], batch["mask"], batch["chain"], batch["resseq"])
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), batch["target"], reduction="none"
        )
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == steps:
            held = evaluate(model, test_batch)
            history.append({"step": step, **held})
            print(
                f"  {variant} seed {seed} step {step:4d}  "
                f"rec {held['recovery']:.3f}  iface {held['interface']:.3f}  nll {held['nll']:.3f}",
                flush=True,
            )
    final = history[-1]
    matched = next(row for row in history if row["step"] == min(STEPS, steps))
    return {"final": final, "matched": matched, "history": history, "steps": steps}


def main() -> None:
    cache = ROOT / "complexes.json"
    if not cache.exists():
        print("fetching heterodimers", flush=True)
        build_cache(24)
    complexes = load_cache()
    split_path = ROOT / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
    else:
        split = split_clusters(cluster(complexes))
        split_path.write_text(json.dumps(split, indent=2))
    train = [complexes[i] for i in split["train"]]
    test = [complexes[i] for i in split["test"]]
    print(f"{len(train)} train  {len(test)} test  ids {[c['id'] for c in test]}", flush=True)
    test_batch = tensor_batch(test)
    iface = int((test_batch["mask"] & test_batch["interface"]).sum())
    total = int(test_batch["mask"].sum())
    print(f"test residues {total}  interface {iface}", flush=True)

    widths = hidden_for(VARIANTS)
    print("widths", widths, flush=True)
    runs = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        steps = STEPS * (2 if variant in DIAGNOSTIC_VARIANTS else 1)
        for seed in SEEDS:
            runs[variant].append(train_one(variant, widths[variant], seed, train, test_batch, steps))

    summary = {}
    for variant, rows in runs.items():
        def col(key: str, which: str = "matched") -> list[float]:
            return [r[which][key] for r in rows]

        summary[variant] = {
            "recovery_mean": sum(col("recovery")) / len(rows),
            "recovery_std": _std(col("recovery")),
            "interface_mean": sum(col("interface")) / len(rows),
            "interface_std": _std(col("interface")),
            "nll_mean": sum(col("nll")) / len(rows),
            "nll_std": _std(col("nll")),
            "seeds": [r["matched"] for r in rows],
        }
        if variant in DIAGNOSTIC_VARIANTS:
            summary[variant]["diagnostic_recovery_mean"] = sum(col("recovery", "final")) / len(rows)
            summary[variant]["diagnostic_interface_mean"] = sum(col("interface", "final")) / len(rows)
            summary[variant]["diagnostic_steps"] = rows[0]["steps"]
        print(
            f"{variant}  recovery {summary[variant]['recovery_mean']:.3f} ± {summary[variant]['recovery_std']:.3f}"
            f"  interface {summary[variant]['interface_mean']:.3f} ± {summary[variant]['interface_std']:.3f}",
            flush=True,
        )

    out = {
        "task": "backbone-only inverse folding on heterodimers",
        "pm_definition": "sample standard deviation across seeds, n-1 denominator",
        "steps_matched": STEPS,
        "seeds": list(SEEDS),
        "split": split,
        "test_ids": [c["id"] for c in test],
        "train_ids": [c["id"] for c in train],
        "summary": summary,
        "note": "Comparisons use the matched budget. D's longer run is diagnostic only.",
    }
    path = Path(__file__).resolve().parent / "results" / "inverse.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
