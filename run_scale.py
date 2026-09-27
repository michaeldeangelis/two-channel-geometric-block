"""C versus A on new heterodimer families.

Checkpoint: lowest unweighted mean of per-complex validation NLL.
The Stage 3 test ids are in train only. S and E are not run.
D is a control, not the comparison. Z is C with coordinates removed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from inverse_fold import (
    AA,
    InverseFolder,
    frequency_scores,
    hidden_for,
    loss_and_scores,
    parameter_count,
    prediction_mix,
    tensor_batch,
    train_background,
)
from proteins import (
    DEV_IDS,
    ROOT,
    audit_partitions,
    build_cache,
    cluster,
    load_cache,
    split_three,
)

STEPS = 400
EVERY = 50
SEEDS = (0, 1, 2)
BATCH = 2
LIMIT = 72


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    mean = _mean(xs)
    return (sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5


def _rows(model: InverseFolder, complexes: list[dict]) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for comp in complexes:
            batch = tensor_batch([comp])
            logits = model.logits(batch["pos"], batch["mask"], batch["chain"], batch["resseq"])
            row = loss_and_scores(logits, batch)
            mix = prediction_mix(logits, batch)
            row["id"] = comp["id"]
            row["argmax_freq"] = mix["argmax_freq"]
            row["mean_softmax"] = mix["mean_softmax"]
            rows.append(row)
    return rows


def _macro(rows: list[dict], key: str) -> float:
    return _mean([r[key] for r in rows])


def _mix(rows: list[dict], key: str) -> list[float]:
    total = sum(r["n"] for r in rows) or 1
    acc = [0.0] * len(AA)
    for row in rows:
        for i, value in enumerate(row[key]):
            acc[i] += value * row["n"]
    return [v / total for v in acc]


def _slim(rows: list[dict]) -> list[dict]:
    keep = (
        "id", "n", "n_interface", "nll", "recovery", "interface", "interface_nll",
        "noninterface", "noninterface_nll",
    )
    return [{k: row[k] for k in keep} for row in rows]


def train_one(name: str, variant: str, hidden: int, geometry: bool, seed: int, train, val, test) -> dict:
    torch.manual_seed(seed)
    model = InverseFolder(variant, hidden, geometry=geometry)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order = list(range(len(train)))
    history = []
    best_score = float("inf")
    best_state = None
    best_step = 1
    for step in range(1, STEPS + 1):
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
        if step == 1 or step % EVERY == 0 or step == STEPS:
            train_rows = _rows(model, train)
            val_rows = _rows(model, val)
            train_nll = _macro(train_rows, "nll")
            val_nll = _macro(val_rows, "nll")
            row = {
                "step": step,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "val_recovery": _macro(val_rows, "recovery"),
                "val_interface_nll": _macro(val_rows, "interface_nll"),
                "val_argmax_freq": _mix(val_rows, "argmax_freq"),
            }
            history.append(row)
            if val_nll < best_score:
                best_score = val_nll
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  {name} seed {seed} step {step:4d}  "
                f"train {train_nll:.3f}  val {val_nll:.3f}  rec {_macro(val_rows, 'recovery'):.3f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    test_rows = _rows(model, test)
    chosen = next(row for row in history if row["step"] == best_step)
    final = history[-1]
    return {
        "best_step": best_step,
        "best_val_nll": best_score,
        "history": [{k: v for k, v in row.items() if k != "val_argmax_freq"} | {"val_argmax_top": AA[max(range(len(AA)), key=lambda i: row["val_argmax_freq"][i])]} for row in history],
        "train_nll_at_best": chosen["train_nll"],
        "final_train_nll": final["train_nll"],
        "final_val_nll": final["val_nll"],
        "test": _slim(test_rows),
        "test_argmax_freq": _mix(test_rows, "argmax_freq"),
        "params": parameter_count(model),
    }


def _paired(test_rows: dict[str, list[dict]]) -> list[dict]:
    """One row per test complex. Metrics are means across seeds."""
    ids = [r["id"] for r in test_rows["A"][0]]
    out = []
    for i, pdb_id in enumerate(ids):
        def col(name: str, key: str) -> float:
            return _mean([seed[i][key] for seed in test_rows[name]])

        out.append({
            "id": pdb_id,
            "n": test_rows["A"][0][i]["n"],
            "n_interface": test_rows["A"][0][i]["n_interface"],
            "A_nll": col("A", "nll"),
            "C_nll": col("C", "nll"),
            "Z_nll": col("Z", "nll"),
            "D_nll": col("D", "nll"),
            "F_nll": test_rows["F"][i]["nll"],
            "A_recovery": col("A", "recovery"),
            "C_recovery": col("C", "recovery"),
            "Z_recovery": col("Z", "recovery"),
            "D_recovery": col("D", "recovery"),
            "F_recovery": test_rows["F"][i]["recovery"],
            "A_interface_nll": col("A", "interface_nll"),
            "C_interface_nll": col("C", "interface_nll"),
            "Z_interface_nll": col("Z", "interface_nll"),
            "F_interface_nll": test_rows["F"][i]["interface_nll"],
            "C_minus_A_nll": col("C", "nll") - col("A", "nll"),
            "C_minus_A_recovery": col("C", "recovery") - col("A", "recovery"),
            "C_minus_A_interface_nll": col("C", "interface_nll") - col("A", "interface_nll"),
        })
    return out


def main() -> None:
    cache = ROOT / "complexes.json"
    have = len(load_cache()) if cache.exists() else 0
    if have < LIMIT:
        print(f"cache has {have}, fetching {LIMIT}", flush=True)
        build_cache(LIMIT, update_split=False)
    complexes = load_cache()
    groups = cluster(complexes)
    split = split_three(complexes, groups)
    audit = audit_partitions(complexes, split)
    print(
        f"clusters {len(groups)}  train {len(split['train'])}  "
        f"val {len(split['val'])}  test {len(split['test'])}  "
        f"worst identity {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    if not audit["passed"]:
        raise SystemExit(f"homology audit failed: {audit['violations'][:5]}")
    if min(len(split["val"]), len(split["test"])) < 8:
        raise SystemExit(
            f"need at least 8 val and 8 test complexes, got {len(split['val'])} and {len(split['test'])}"
        )
    missing = [pdb_id for pdb_id in DEV_IDS if pdb_id not in {c["id"] for c in complexes}]
    if missing:
        raise SystemExit(f"stage 3 test ids missing from the cache: {missing}")
    (ROOT / "split_scale.json").write_text(json.dumps({"split": split, "audit": audit}, indent=2))
    train = [complexes[i] for i in split["train"]]
    val = [complexes[i] for i in split["val"]]
    test = [complexes[i] for i in split["test"]]
    print("val", [c["id"] for c in val], flush=True)
    print("test", [c["id"] for c in test], flush=True)
    background = train_background(train)
    freq = background["freq"]
    print(
        f"train mode {background['mode_aa']} freq {float(freq.max()):.3f}",
        flush=True,
    )

    widths = hidden_for(("A", "C", "D"))
    specs = {
        "A": ("A", widths["A"], True),
        "C": ("C", widths["C"], True),
        "Z": ("C", widths["C"], False),
        "D": ("D", widths["D"], True),
    }
    print(
        "widths",
        {k: v[1] for k, v in specs.items()},
        "params",
        {k: parameter_count(InverseFolder(v, h, geometry=g)) for k, (v, h, g) in specs.items()},
        flush=True,
    )

    runs = {name: [] for name in specs}
    test_rows: dict[str, list] = {name: [] for name in specs}
    for name, (variant, hidden, geometry) in specs.items():
        for seed in SEEDS:
            done = train_one(name, variant, hidden, geometry, seed, train, val, test)
            test_rows[name].append(done.pop("test"))
            runs[name].append(done)

    freq_rows = []
    for comp in test:
        batch = tensor_batch([comp])
        row = frequency_scores(batch, background)
        row["id"] = comp["id"]
        freq_rows.append(_slim([row])[0])
    test_rows["F"] = freq_rows
    paired = _paired(test_rows)

    def seed_macro(name: str, key: str) -> list[float]:
        return [_mean([row[key] for row in seed]) for seed in test_rows[name]]

    summary = {}
    for name in specs:
        nlls = seed_macro(name, "nll")
        recs = seed_macro(name, "recovery")
        summary[name] = {
            "params": runs[name][0]["params"],
            "hidden": specs[name][1],
            "geometry": specs[name][2],
            "test_nll_mean_across_seeds": _mean(nlls),
            "test_nll_seed_std": _std(nlls),
            "test_recovery_mean_across_seeds": _mean(recs),
            "test_recovery_seed_std": _std(recs),
            "best_steps": [r["best_step"] for r in runs[name]],
            "train_change_after_best": [
                r["final_train_nll"] - r["train_nll_at_best"] for r in runs[name]
            ],
            "val_change_after_best": [
                r["final_val_nll"] - r["best_val_nll"] for r in runs[name]
            ],
        }
        print(
            f"{name}  test nll {_mean(nlls):.3f} ± {_std(nlls):.3f}"
            f"  recovery {_mean(recs):.3f} ± {_std(recs):.3f}",
            flush=True,
        )

    nll_delta = [r["C_minus_A_nll"] for r in paired]
    rec_delta = [r["C_minus_A_recovery"] for r in paired]
    iface_delta = [r["C_minus_A_interface_nll"] for r in paired]
    print(
        f"paired C-A nll {_mean(nll_delta):.3f}  "
        f"complexes C better {sum(d < 0 for d in nll_delta)}/{len(paired)}",
        flush=True,
    )
    out = {
        "task": "C versus A on heterodimers, checkpointed on validation NLL",
        "primary_unit": "held-out complex, unweighted",
        "seed_std_means": "training variability across seeds, not uncertainty across proteins",
        "checkpoint": "lowest unweighted mean of per-complex validation NLL",
        "retired_stage3_test_ids": list(DEV_IDS),
        "geometry_ablation": "Z is C with all coordinates set to zero. Chain id and sequence index remain.",
        "steps": STEPS,
        "seeds": list(SEEDS),
        "widths": {k: v[1] for k, v in specs.items()},
        "train_ids": [c["id"] for c in train],
        "val_ids": [c["id"] for c in val],
        "test_ids": [c["id"] for c in test],
        "audit": audit,
        "train_mode_aa": background["mode_aa"],
        "train_freq": [float(x) for x in freq],
        "summary": summary,
        "paired": paired,
        "frequency_test": freq_rows,
        "runs": runs,
    }
    path = Path(__file__).resolve().parent / "results" / "scale.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
