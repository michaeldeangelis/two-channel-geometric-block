"""Does gate mass recover the gated sum?

C is the implemented sum. N is the same messages divided by gate mass.
Q is that mean with log(1+rho) restored. Previous val/test ids stay out
of this test. New complexes supply the locked split.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_fold import InverseFolder, parameter_count
from proteins import ROOT, audit_partitions, build_cache, cluster, load_cache
from run_confirm import cluster_interval
from run_scale import _mean, train_one

LIMIT = 480
WIDTH = 80
EXPECTED = 22278
VARIANTS = ("C", "N", "Q")
LOCK = ROOT / "split_magnitude.json"
RESULTS = Path(__file__).resolve().parent / "results"


def _ids(path: Path, *keys: str) -> set[str]:
    payload = json.loads(path.read_text())
    found = set()
    for key in keys:
        found.update(payload[key])
    return found


def previously_scored() -> set[str]:
    scored = _ids(RESULTS / "confirm.json", "val_ids", "test_ids", "development_ids")
    scored |= _ids(RESULTS / "scale.json", "val_ids", "test_ids", "retired_stage3_test_ids")
    return scored


def split_fresh(complexes, clusters, forbidden: set[str], fresh: set[str], n_test: int = 30, n_val: int = 24):
    train: list[int] = []
    flexible: list[list[int]] = []
    for group in clusters:
        ids = [complexes[i]["id"] for i in group]
        untouched = all(pdb_id in fresh for pdb_id in ids) and not any(pdb_id in forbidden for pdb_id in ids)
        if untouched:
            flexible.append(group)
        else:
            train.extend(group)
    test: list[int] = []
    val: list[int] = []
    for group in sorted(flexible, key=len):
        if len(test) < n_test:
            test.extend(group)
        elif len(val) < n_val:
            val.extend(group)
        else:
            train.extend(group)
    return {"train": train, "val": val, "test": test}


def _enough(split: dict) -> bool:
    return len(split["test"]) >= 20 and len(split["val"]) >= 12


def _lock(complexes: list[dict], fresh: set[str]) -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    forbidden = previously_scored()
    groups = cluster(complexes)
    split = split_fresh(complexes, groups, forbidden, fresh, n_test=30, n_val=20)
    if not _enough(split):
        split = split_fresh(complexes, groups, forbidden, fresh, n_test=24, n_val=12)
    if not _enough(split):
        raise SystemExit(
            f"not enough new clusters: test {len(split['test'])} val {len(split['val'])} fresh {len(fresh)}"
        )
    audit = audit_partitions(complexes, split)
    if not audit["passed"]:
        raise SystemExit(f"homology audit failed: {audit['violations'][:5]}")
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
        "forbidden_ids": sorted(forbidden),
        "audit": audit,
        "n_clusters": len(groups),
        "n_fresh": len(fresh),
    }
    leaked = [pdb_id for pdb_id in lock["test"] + lock["val"] if pdb_id in forbidden]
    if leaked:
        raise SystemExit(f"previously scored ids in val or test: {leaked[:8]}")
    LOCK.write_text(json.dumps(lock, indent=2))
    print(
        f"locked train {len(lock['train'])} val {len(lock['val'])} test {len(lock['test'])} "
        f"fresh {len(fresh)} clusters {len(groups)} worst {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    return lock


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    missing = [pdb_id for pdb_id in ids if pdb_id not in by_id]
    if missing:
        raise SystemExit(f"locked ids missing: {missing[:8]}")
    return [by_id[pdb_id] for pdb_id in ids]


def main() -> None:
    for name in VARIANTS:
        count = parameter_count(InverseFolder(name, WIDTH))
        if count != EXPECTED:
            raise SystemExit(f"{name} has {count} parameters, expected {EXPECTED}")
    before = {c["id"] for c in load_cache()} if (ROOT / "complexes.json").exists() else set()
    if not LOCK.exists() and len(before) < LIMIT:
        print(f"cache has {len(before)}, fetching {LIMIT}", flush=True)
        build_cache(LIMIT, update_split=False)
    complexes = load_cache()
    fresh = {c["id"] for c in complexes if c["id"] not in before}
    if LOCK.exists():
        fresh = set()
    lock = _lock(complexes, fresh if not LOCK.exists() else set(json.loads(LOCK.read_text()).get("fresh_ids", [])))
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    print("test", lock["test"], flush=True)

    runs = {}
    test_rows = {}
    for name in VARIANTS:
        runs[name] = []
        test_rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(name, name, WIDTH, True, seed, train, val, test)
            test_rows[name].append(done.pop("test"))
            runs[name].append(done)

    paired = []
    for i, comp in enumerate(test):
        def col(name: str, key: str, i: int = i) -> float:
            return _mean([seed[i][key] for seed in test_rows[name]])

        paired.append({
            "id": comp["id"],
            "cluster": lock["test_cluster"][comp["id"]],
            "C_nll": col("C", "nll"),
            "N_nll": col("N", "nll"),
            "Q_nll": col("Q", "nll"),
            "N_minus_C_nll": col("N", "nll") - col("C", "nll"),
            "Q_minus_C_nll": col("Q", "nll") - col("C", "nll"),
            "Q_minus_N_nll": col("Q", "nll") - col("N", "nll"),
            "C_recovery": col("C", "recovery"),
            "N_recovery": col("N", "recovery"),
            "Q_recovery": col("Q", "recovery"),
            "C_interface_nll": col("C", "interface_nll"),
            "N_interface_nll": col("N", "interface_nll"),
            "Q_interface_nll": col("Q", "interface_nll"),
        })

    def groups(key: str):
        buckets: dict[str, list[float]] = {}
        for row in paired:
            buckets.setdefault(row["cluster"], []).append(row[key])
        return list(buckets.values())

    effects = {
        key: cluster_interval(groups(key))
        for key in ("N_minus_C_nll", "Q_minus_C_nll", "Q_minus_N_nll")
    }
    for key, estimate in effects.items():
        print(
            f"{key} {estimate['point']:+.3f} [{estimate['lo']:+.3f}, {estimate['hi']:+.3f}] "
            f"clusters {estimate['n_clusters']}",
            flush=True,
        )
    out = {
        "task": "gated sum versus normalized gated mean, with and without gate mass",
        "eps": 1e-8,
        "mass_feature": "log(1+rho), same slot C already uses; N writes zero there",
        "primary_endpoint": "whole-complex NLL",
        "uncertainty": "95% cluster bootstrap, seeds averaged per complex before resampling; not retraining variation",
        "width": WIDTH,
        "params": EXPECTED,
        "train_ids": lock["train"],
        "val_ids": lock["val"],
        "test_ids": lock["test"],
        "audit": lock["audit"],
        "n_clusters_in_cache": lock["n_clusters"],
        "effects": effects,
        "paired": paired,
        "runs": runs,
    }
    path = RESULTS / "magnitude.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
