"""Frame-edge inverse folding. Sum versus mean-plus-mass, after a composition gate.

Depth stays 3. The new information is the backbone frame on each edge.
P is the gated sum with those geometric channels zeroed and the 10 Å
graph left in place. Earlier validation and test complexes stay out.
"""

from __future__ import annotations

import json
from pathlib import Path

from frames import attach_backbone
from inverse_fold import FrameFolder, frequency_scores, parameter_count, train_background
from proteins import ROOT, audit_partitions, cluster, load_cache
from run_confirm import cluster_interval
from run_deep import LOCK as DEEP_LOCK
from run_magnitude import LOCK as MAG_LOCK
from run_magnitude import previously_scored, split_fresh
from run_scale import _mean, train_one

WIDTH = 128
LOCK = ROOT / "split_strong.json"
RESULTS = Path(__file__).resolve().parent / "results"


def build_sum(variant: str, hidden: int, geometry: bool = True) -> FrameFolder:
    return FrameFolder(variant, hidden, structure=True)


def build_plain(variant: str, hidden: int, geometry: bool = True) -> FrameFolder:
    return FrameFolder("C", hidden, structure=False)


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    return [by_id[pdb_id] for pdb_id in ids]


def _forbidden() -> set[str]:
    blocked = previously_scored()
    for path in (MAG_LOCK, DEEP_LOCK):
        payload = json.loads(path.read_text())
        blocked |= set(payload["val"])
        blocked |= set(payload["test"])
    return blocked


def _lock(complexes) -> dict:
    if LOCK.exists():
        return json.loads(LOCK.read_text())
    forbidden = _forbidden()
    groups = cluster(complexes)
    present = {c["id"] for c in complexes}
    fresh = present - forbidden
    split = split_fresh(complexes, groups, forbidden, fresh, n_test=30, n_val=20)
    if len(split["test"]) < 20 or len(split["val"]) < 12:
        split = split_fresh(complexes, groups, forbidden, fresh, n_test=24, n_val=12)
    if len(split["test"]) < 20 or len(split["val"]) < 12:
        raise SystemExit(f"strong test would have test {len(split['test'])} val {len(split['val'])}")
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
    }
    LOCK.write_text(json.dumps(lock, indent=2))
    print(
        f"locked train {len(lock['train'])} val {len(lock['val'])} test {len(lock['test'])} "
        f"worst {audit['worst_cross_identity']:.3f}",
        flush=True,
    )
    return lock


def _effect(rows: list[dict], key: str) -> dict:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["cluster"], []).append(row[key])
    return cluster_interval(list(buckets.values()))


def main() -> None:
    full = parameter_count(build_sum("C", WIDTH))
    other = parameter_count(build_sum("Q", WIDTH))
    plain = parameter_count(build_plain("C", WIDTH))
    if not (full == other == plain):
        raise SystemExit(f"param mismatch C {full} Q {other} P {plain}")
    print("params", full, flush=True)
    raw = load_cache()
    complexes = []
    missed = 0
    for comp in raw:
        if attach_backbone(comp):
            complexes.append(comp)
        else:
            missed += 1
    print(f"backbone {len(complexes)} missing {missed}", flush=True)
    lock = _lock(complexes)
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    background = train_background(train)
    runs = {}
    rows = {}
    jobs = (("C", build_sum), ("Q", build_sum), ("P", build_plain))
    for name, maker in jobs:
        variant = "Q" if name == "Q" else "C"
        runs[name] = []
        rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(name, variant, WIDTH, True, seed, train, val, test, build=maker)
            rows[name].append(done.pop("test"))
            runs[name].append({k: v for k, v in done.items() if k != "model"})
    freq_rows = []
    # Frequency baseline has no weights. Score each test complex once.
    from inverse_fold import tensor_batch

    for comp in test:
        batch = tensor_batch([comp])
        scored = frequency_scores(batch, background)
        freq_rows.append({"id": comp["id"], "nll": scored["nll"], "recovery": scored["recovery"]})
    paired = []
    for i, pdb_id in enumerate(lock["test"]):
        def col(name: str, key: str, i: int = i) -> float:
            return _mean([seed[i][key] for seed in rows[name]])

        paired.append({
            "id": pdb_id,
            "cluster": lock["test_cluster"][pdb_id],
            "C_nll": col("C", "nll"),
            "Q_nll": col("Q", "nll"),
            "P_nll": col("P", "nll"),
            "F_nll": freq_rows[i]["nll"],
            "C_minus_F_nll": col("C", "nll") - freq_rows[i]["nll"],
            "C_minus_P_nll": col("C", "nll") - col("P", "nll"),
            "Q_minus_C_nll": col("Q", "nll") - col("C", "nll"),
            "C_recovery": col("C", "recovery"),
            "Q_recovery": col("Q", "recovery"),
            "F_recovery": freq_rows[i]["recovery"],
            "C_interface_nll": col("C", "interface_nll"),
            "Q_interface_nll": col("Q", "interface_nll"),
        })
    effect = {
        "C_minus_F": _effect(paired, "C_minus_F_nll"),
        "C_minus_P": _effect(paired, "C_minus_P_nll"),
        "Q_minus_C": _effect(paired, "Q_minus_C_nll"),
    }
    for key, item in effect.items():
        print(
            f"{key} {item['point']:+.3f} [{item['lo']:+.3f}, {item['hi']:+.3f}] clusters {item['n_clusters']}",
            flush=True,
        )
    out = {
        "task": "frame-edge inverse folding, sum versus mean-plus-mass, composition gate first",
        "depth": 3,
        "edge": "16-bin CA RBF, local CA offset, relative N-CA-C rotation, sequence separation, chain identity",
        "graph": "10 A CA window, unchanged when geometric channels are zeroed",
        "hidden": WIDTH,
        "params": full,
        "steps": 400,
        "uncertainty": "95% cluster bootstrap, seeds averaged first; not retraining variation",
        "train_ids": lock["train"],
        "val_ids": lock["val"],
        "test_ids": lock["test"],
        "audit": lock["audit"],
        "backbone_kept": len(complexes),
        "backbone_missing": missed,
        "mode_aa": background["mode_aa"],
        "effect": effect,
        "paired": paired,
        "runs": runs,
    }
    path = RESULTS / "strong.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
