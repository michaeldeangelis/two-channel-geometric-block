"""Sum versus mean-plus-mass on new families, with frame edges.

The step budget is 800 because the stage-9 validation selected step 400
for every seed. That validation is the checkpoint set. The new test is
not used to stop.
"""

from __future__ import annotations

import json
from pathlib import Path

from frames import attach_backbone
from inverse_fold import FrameFolder, frequency_scores, parameter_count, tensor_batch, train_background
from proteins import ROOT, audit_partitions, cluster, extend_cache, load_cache
from run_confirm import cluster_interval
from run_magnitude import previously_scored
from run_scale import _mean, train_one
from run_strong import LOCK as STRONG_LOCK

WIDTH = 128
STEPS = 800
N_CLUSTERS = 24
MIN_CLUSTERS = 15
DATE_FROM = "2022-06-02"
DATE_TO = "2026-09-01"
SNAP = ROOT / "ids_before_families.json"
LOCK = ROOT / "split_families.json"
RESULTS = Path(__file__).resolve().parent / "results"


def build(variant: str, hidden: int, geometry: bool = True) -> FrameFolder:
    return FrameFolder(variant, hidden, structure=True)


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    missing = [pdb_id for pdb_id in ids if pdb_id not in by_id]
    if missing:
        raise SystemExit(f"missing complexes {missing[:5]}")
    return [by_id[pdb_id] for pdb_id in ids]


def _forbidden() -> set[str]:
    blocked = previously_scored()
    for name in ("split_magnitude.json", "split_deep.json", "split_strong.json"):
        payload = json.loads((ROOT / name).read_text())
        blocked |= set(payload["val"])
        blocked |= set(payload["test"])
    return blocked


def _with_backbone(complexes):
    kept = []
    missed = 0
    for comp in complexes:
        if attach_backbone(comp):
            kept.append(comp)
        else:
            missed += 1
    return kept, missed


def _eligible(complexes, groups, new_ids, forbidden):
    found = []
    for group in groups:
        ids = [complexes[i]["id"] for i in group]
        if all(pdb_id in new_ids for pdb_id in ids) and not any(pdb_id in forbidden for pdb_id in ids):
            found.append(group)
    return sorted(found, key=len)


def _prepare(complexes, new_ids) -> dict:
    forbidden = _forbidden()
    groups = cluster(complexes)
    eligible = _eligible(complexes, groups, new_ids, forbidden)
    print(f"eligible new clusters {len(eligible)}", flush=True)
    if len(eligible) < MIN_CLUSTERS:
        raise SystemExit(f"only {len(eligible)} new clusters; no training run")
    strong = json.loads(STRONG_LOCK.read_text())
    val_seed = set(strong["val"])
    val_ids = set()
    for group in groups:
        ids = [complexes[i]["id"] for i in group]
        if any(pdb_id in val_seed for pdb_id in ids):
            val_ids.update(ids)
    test_groups = eligible[:N_CLUSTERS]
    test_ids = [complexes[i]["id"] for group in test_groups for i in group]
    if set(test_ids) & val_ids:
        raise SystemExit("new test collided with the development validation")
    cluster_of = {}
    for group in groups:
        key = "|".join(sorted(complexes[i]["id"] for i in group))
        for i in group:
            cluster_of[complexes[i]["id"]] = key
    by_id = {c["id"]: i for i, c in enumerate(complexes)}
    split = {
        "train": [by_id[c["id"]] for c in complexes if c["id"] not in val_ids and c["id"] not in set(test_ids)],
        "val": [by_id[pdb_id] for pdb_id in val_ids if pdb_id in by_id],
        "test": [by_id[pdb_id] for pdb_id in test_ids],
    }
    audit = audit_partitions(complexes, split)
    if not audit["passed"]:
        raise SystemExit(audit["violations"][:5])
    lock = {
        "train": [complexes[i]["id"] for i in split["train"]],
        "val": [complexes[i]["id"] for i in split["val"]],
        "test": test_ids,
        "test_cluster": {pdb_id: cluster_of[pdb_id] for pdb_id in test_ids},
        "test_cluster_sizes": sorted(len(group) for group in test_groups),
        "n_test_clusters": len(test_groups),
        "audit": audit,
        "steps": STEPS,
        "resolution_note": "2.5, widened to 3.0 only if fewer than 15 clusters",
    }
    LOCK.write_text(json.dumps(lock, indent=2))
    print(
        f"locked train {len(lock['train'])} val {len(lock['val'])} "
        f"test {len(lock['test'])} clusters {lock['n_test_clusters']} "
        f"worst {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    return lock


def _effect(rows, key):
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["cluster"], []).append(row[key])
    return cluster_interval(list(buckets.values()))


def main() -> None:
    count = parameter_count(build("C", WIDTH))
    other = parameter_count(build("Q", WIDTH))
    if count != other:
        raise SystemExit(f"param mismatch {count} {other}")
    if not SNAP.exists():
        SNAP.write_text(json.dumps(sorted(c["id"] for c in load_cache())))
    old_ids = set(json.loads(SNAP.read_text()))
    if not LOCK.exists():
        extend_cache(300, DATE_FROM, DATE_TO, 2.5)
        pooled, missed = _with_backbone(load_cache())
        groups = cluster(pooled)
        n_ok = len(_eligible(pooled, groups, {c["id"] for c in pooled} - old_ids, _forbidden()))
        print(f"at 2.5 A: eligible {n_ok} missing backbone {missed}", flush=True)
        if n_ok < MIN_CLUSTERS:
            extend_cache(300, DATE_FROM, DATE_TO, 3.0)
    else:
        print("reusing family lock", flush=True)
    pooled, missed = _with_backbone(load_cache())
    new_ids = {c["id"] for c in pooled} - old_ids
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else _prepare(pooled, new_ids)
    train = _take(pooled, lock["train"])
    val = _take(pooled, lock["val"])
    test = _take(pooled, lock["test"])
    background = train_background(train)
    runs = {}
    rows = {}
    for name in ("C", "Q"):
        runs[name] = []
        rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(name, name, WIDTH, True, seed, train, val, test, build=build, steps=STEPS)
            rows[name].append(done.pop("test"))
            runs[name].append({k: v for k, v in done.items() if k != "model"})
    freq = []
    for comp in test:
        scored = frequency_scores(tensor_batch([comp]), background)
        freq.append(scored["nll"])
    paired = []
    for i, pdb_id in enumerate(lock["test"]):
        def col(name: str, key: str, i: int = i) -> float:
            return _mean([seed[i][key] for seed in rows[name]])

        paired.append({
            "id": pdb_id,
            "cluster": lock["test_cluster"][pdb_id],
            "C_nll": col("C", "nll"),
            "Q_nll": col("Q", "nll"),
            "F_nll": freq[i],
            "Q_minus_C_nll": col("Q", "nll") - col("C", "nll"),
            "C_minus_F_nll": col("C", "nll") - freq[i],
        })
    cluster_gap = _effect(paired, "Q_minus_C_nll")
    complex_gap = _mean([row["Q_minus_C_nll"] for row in paired])
    print(
        f"Q-C cluster {cluster_gap['point']:+.3f} [{cluster_gap['lo']:+.3f}, {cluster_gap['hi']:+.3f}] "
        f"complex {complex_gap:+.3f}",
        flush=True,
    )
    out = {
        "task": "frame-edge sum versus mean-plus-mass on new families",
        "steps": STEPS,
        "budget_reason": "stage-9 validation selected step 400 for every C and Q seed",
        "weighting": "cluster effect gives each cluster one vote; complex effect gives each complex one vote",
        "hidden": WIDTH,
        "params": count,
        "backbone_missing": missed,
        "n_test_clusters": lock["n_test_clusters"],
        "test_cluster_sizes": lock["test_cluster_sizes"],
        "uncertainty": "95% cluster bootstrap, seeds averaged first; not retraining variation",
        "train_ids": lock["train"],
        "val_ids": lock["val"],
        "test_ids": lock["test"],
        "audit": lock["audit"],
        "cluster_effect": cluster_gap,
        "complex_effect": complex_gap,
        "frequency_effect": _effect(paired, "C_minus_F_nll"),
        "frequency_complex_effect": _mean([row["C_minus_F_nll"] for row in paired]),
        "paired": paired,
        "runs": runs,
    }
    path = RESULTS / "families.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
