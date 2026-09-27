"""Does the sum still beat mean-plus-mass inside a deeper encoder?

Three blocks, width 64, mixer hidden 128. C versus Q only.
Earlier validation and test complexes stay out of this test.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_fold import StackedFolder, parameter_count
from proteins import ROOT, audit_partitions, cluster, load_cache
from run_confirm import cluster_interval
from run_magnitude import previously_scored, split_fresh
from run_scale import _mean, train_one

WIDTH = 128
DEPTH = 3
LOCK = ROOT / "split_deep.json"
RESULTS = Path(__file__).resolve().parent / "results"


def build(variant: str, hidden: int, geometry: bool = True) -> StackedFolder:
    return StackedFolder(variant, hidden, geometry=geometry, depth=DEPTH, h_dim=64, msg=64)


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    return [by_id[pdb_id] for pdb_id in ids]


def _lock(complexes) -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    forbidden = previously_scored()
    forbidden |= set(json.loads((ROOT / "split_magnitude.json").read_text())["val"])
    forbidden |= set(json.loads((ROOT / "split_magnitude.json").read_text())["test"])
    groups = cluster(complexes)
    present = {c["id"] for c in complexes}
    fresh = present - forbidden
    split = split_fresh(complexes, groups, forbidden, fresh, n_test=30, n_val=20)
    if len(split["test"]) < 20:
        split = split_fresh(complexes, groups, forbidden, fresh, n_test=24, n_val=12)
    if len(split["test"]) < 20:
        raise SystemExit(f"deep test would have only {len(split['test'])} complexes")
    audit = audit_partitions(complexes, split)
    if not audit["passed"]:
        raise SystemExit(audit["violations"][:5])
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
        "audit": audit,
        "n_clusters": len(groups),
    }
    LOCK.write_text(json.dumps(lock, indent=2))
    print(
        f"locked train {len(lock['train'])} val {len(lock['val'])} test {len(lock['test'])} "
        f"worst {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    return lock


def main() -> None:
    count = parameter_count(build("C", WIDTH))
    other = parameter_count(build("Q", WIDTH))
    if count != other:
        raise SystemExit(f"param mismatch C {count} Q {other}")
    print("params", count, "depth", DEPTH, flush=True)
    complexes = load_cache()
    lock = _lock(complexes)
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    runs = {}
    rows = {}
    for name in ("C", "Q"):
        runs[name] = []
        rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(name, name, WIDTH, True, seed, train, val, test, build=build)
            rows[name].append(done.pop("test"))
            runs[name].append({k: v for k, v in done.items() if k != "model"})
    paired = []
    for i, pdb_id in enumerate(lock["test"]):
        def col(name: str, key: str, i: int = i) -> float:
            return _mean([seed[i][key] for seed in rows[name]])

        paired.append({
            "id": pdb_id,
            "cluster": lock["test_cluster"][pdb_id],
            "C_nll": col("C", "nll"),
            "Q_nll": col("Q", "nll"),
            "Q_minus_C_nll": col("Q", "nll") - col("C", "nll"),
            "C_recovery": col("C", "recovery"),
            "Q_recovery": col("Q", "recovery"),
            "C_interface_nll": col("C", "interface_nll"),
            "Q_interface_nll": col("Q", "interface_nll"),
        })
    buckets: dict[str, list[float]] = {}
    for row in paired:
        buckets.setdefault(row["cluster"], []).append(row["Q_minus_C_nll"])
    effect = cluster_interval(list(buckets.values()))
    print(
        f"Q-C nll {effect['point']:+.3f} [{effect['lo']:+.3f}, {effect['hi']:+.3f}] "
        f"clusters {effect['n_clusters']}",
        flush=True,
    )
    out = {
        "task": "sum versus mean-plus-mass in a 3-block width-64 encoder",
        "depth": DEPTH,
        "h_dim": 64,
        "msg": 64,
        "hidden": WIDTH,
        "params": count,
        "uncertainty": "95% cluster bootstrap, seeds averaged first; not retraining variation",
        "train_ids": lock["train"],
        "val_ids": lock["val"],
        "test_ids": lock["test"],
        "audit": lock["audit"],
        "effect": effect,
        "paired": paired,
        "runs": runs,
    }
    path = RESULTS / "deep.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
