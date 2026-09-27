"""Locked confirmation. Same C and A as the development screen, more families.

The Stage 5 validation and test ids stay in train. Whole-complex NLL is the
primary endpoint. Uncertainty resamples homology clusters.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from inverse_fold import InverseFolder, frequency_scores, parameter_count, tensor_batch, train_background
from proteins import ROOT, audit_partitions, build_cache, cluster, load_cache, split_three
from run_scale import _mean, _slim, _std, train_one

LIMIT = 240
STEPS_NOTE = 400
WIDTHS = {"A": 104, "C": 80}
EXPECTED_PARAMS = {"A": 23561, "C": 22278}
BOOTSTRAP = 2000
LOCK = ROOT / "split_confirm.json"
SCALE = Path(__file__).resolve().parent / "results" / "scale.json"


def development_ids() -> set[str]:
    scale = json.loads(SCALE.read_text())
    return set(scale["val_ids"]) | set(scale["test_ids"]) | set(scale["retired_stage3_test_ids"])


def cluster_interval(groups: list[list[float]], n: int = BOOTSTRAP, seed: int = 0) -> dict[str, float]:
    """Mean of cluster means, and the 2.5/97.5 percentiles of a cluster bootstrap."""
    point = _mean([_mean(group) for group in groups])
    rng = random.Random(seed)
    m = len(groups)
    draws = []
    for _ in range(n):
        pick = [groups[rng.randrange(m)] for _ in range(m)]
        draws.append(_mean([_mean(group) for group in pick]))
    draws.sort()
    return {
        "point": point,
        "lo": draws[int(0.025 * (n - 1))],
        "hi": draws[int(0.975 * (n - 1))],
        "n_clusters": m,
        "resamples": n,
    }


def _lock_split(complexes: list[dict]) -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    dev = development_ids()
    present = {c["id"] for c in complexes}
    missing = sorted(dev - present)
    if missing:
        raise SystemExit(f"development ids missing from the cache: {missing}")
    groups = cluster(complexes)
    split = split_three(complexes, groups, force_ids=dev)
    audit = audit_partitions(complexes, split)
    if not audit["passed"]:
        raise SystemExit(f"homology audit failed: {audit['violations'][:5]}")
    if len(split["test"]) < 20:
        raise SystemExit(f"locked test would have only {len(split['test'])} complexes")
    cluster_of = {}
    for group in groups:
        key = "|".join(sorted(complexes[i]["id"] for i in group))
        for i in group:
            cluster_of[complexes[i]["id"]] = key
    lock = {
        "train": [complexes[i]["id"] for i in split["train"]],
        "val": [complexes[i]["id"] for i in split["val"]],
        "test": [complexes[i]["id"] for i in split["test"]],
        "test_cluster": {complexes[i]["id"]: cluster_of[complexes[i]["id"]] for i in split["test"]},
        "development_ids": sorted(dev),
        "audit": audit,
        "n_clusters": len(groups),
    }
    LOCK.write_text(json.dumps(lock, indent=2))
    print(
        f"locked train {len(lock['train'])} val {len(lock['val'])} test {len(lock['test'])} "
        f"clusters {lock['n_clusters']} worst identity {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    return lock


def _take(complexes: list[dict], ids: list[str]) -> list[dict]:
    by_id = {c["id"]: c for c in complexes}
    missing = [pdb_id for pdb_id in ids if pdb_id not in by_id]
    if missing:
        raise SystemExit(f"locked ids missing from the cache: {missing[:8]}")
    return [by_id[pdb_id] for pdb_id in ids]


def main() -> None:
    if not LOCK.exists():
        have = len(load_cache()) if (ROOT / "complexes.json").exists() else 0
        if have < LIMIT:
            print(f"cache has {have}, fetching {LIMIT}", flush=True)
            build_cache(LIMIT, update_split=False)
    complexes = load_cache()
    lock = _lock_split(complexes)
    dev = set(lock["development_ids"])
    leaked = [pdb_id for pdb_id in lock["test"] + lock["val"] if pdb_id in dev]
    if leaked:
        raise SystemExit(f"development ids in val or test: {leaked}")
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    print("test", lock["test"], flush=True)
    for name, hidden in WIDTHS.items():
        count = parameter_count(InverseFolder(name, hidden))
        if count != EXPECTED_PARAMS[name]:
            raise SystemExit(f"{name} has {count} parameters, expected {EXPECTED_PARAMS[name]}")
    background = train_background(train)
    print(f"train mode {background['mode_aa']} freq {float(background['freq'].max()):.3f}", flush=True)

    specs = {"A": ("A", WIDTHS["A"], True), "C": ("C", WIDTHS["C"], True)}
    runs = {}
    test_rows = {}
    for name, (variant, hidden, geometry) in specs.items():
        runs[name] = []
        test_rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(name, variant, hidden, geometry, seed, train, val, test)
            test_rows[name].append(done.pop("test"))
            runs[name].append(done)

    freq_rows = []
    for comp in test:
        batch = tensor_batch([comp])
        row = _slim([frequency_scores(batch, background) | {"id": comp["id"]}])[0]
        row["cluster"] = lock["test_cluster"][comp["id"]]
        freq_rows.append(row)

    paired = []
    for i, comp in enumerate(test):
        def col(name: str, key: str, i: int = i) -> float:
            return _mean([seed[i][key] for seed in test_rows[name]])

        paired.append({
            "id": comp["id"],
            "cluster": lock["test_cluster"][comp["id"]],
            "n": test_rows["A"][0][i]["n"],
            "A_nll": col("A", "nll"),
            "C_nll": col("C", "nll"),
            "F_nll": freq_rows[i]["nll"],
            "A_recovery": col("A", "recovery"),
            "C_recovery": col("C", "recovery"),
            "F_recovery": freq_rows[i]["recovery"],
            "A_interface_nll": col("A", "interface_nll"),
            "C_interface_nll": col("C", "interface_nll"),
            "F_interface_nll": freq_rows[i]["interface_nll"],
            "C_minus_A_nll": col("C", "nll") - col("A", "nll"),
            "C_minus_F_nll": col("C", "nll") - freq_rows[i]["nll"],
            "C_minus_A_recovery": col("C", "recovery") - col("A", "recovery"),
            "C_minus_A_interface_nll": col("C", "interface_nll") - col("A", "interface_nll"),
        })

    def groups(key: str) -> list[list[float]]:
        buckets: dict[str, list[float]] = {}
        for row in paired:
            buckets.setdefault(row["cluster"], []).append(row[key])
        return list(buckets.values())

    effects = {
        key: cluster_interval(groups(key))
        for key in (
            "C_minus_A_nll",
            "C_minus_F_nll",
            "C_minus_A_recovery",
            "C_minus_A_interface_nll",
        )
    }
    primary = effects["C_minus_A_nll"]
    print(
        f"cluster C-A nll {primary['point']:.3f}  "
        f"bootstrap [{primary['lo']:.3f}, {primary['hi']:.3f}]  "
        f"clusters {primary['n_clusters']}",
        flush=True,
    )
    out = {
        "task": "locked confirmation of C versus A and the frequency baseline",
        "primary_endpoint": "whole-complex NLL",
        "uncertainty": "cluster bootstrap, 2000 resamples, 2.5 and 97.5 percentiles of the mean of cluster means",
        "widths": WIDTHS,
        "params": EXPECTED_PARAMS,
        "steps": STEPS_NOTE,
        "seeds": [0, 1, 2],
        "train_ids": lock["train"],
        "val_ids": lock["val"],
        "test_ids": lock["test"],
        "development_ids": lock["development_ids"],
        "n_clusters_in_cache": lock["n_clusters"],
        "n_test_clusters": primary["n_clusters"],
        "audit": lock["audit"],
        "train_mode_aa": background["mode_aa"],
        "train_freq_max": float(background["freq"].max()),
        "effects": effects,
        "perplexity_ratio_C_over_A": math.exp(primary["point"]),
        "paired": paired,
        "seed_test_nll": {
            name: {
                "mean": _mean([_mean([row["nll"] for row in seed]) for seed in rows]),
                "seed_std": _std([_mean([row["nll"] for row in seed]) for seed in rows]),
            }
            for name, rows in test_rows.items()
        },
        "runs": runs,
    }
    path = Path(__file__).resolve().parent / "results" / "confirm.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
