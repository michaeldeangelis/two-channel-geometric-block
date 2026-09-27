"""Same deep test, twice the steps. Follow-up on an already examined split.

Both aggregations train for 800 steps. The checkpoint is still the lowest
validation NLL. The test is scored only after that choice.
"""

from __future__ import annotations

import json
from pathlib import Path

from proteins import load_cache
from run_confirm import cluster_interval
from run_deep import DEPTH, LOCK, WIDTH, _take, build
from run_scale import _mean, train_one

STEPS = 800
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    if not LOCK.exists():
        raise SystemExit("deep lock missing; this follow-up does not make a new split")
    lock = json.loads(LOCK.read_text())
    prior = json.loads((RESULTS / "deep.json").read_text())
    if prior["test_ids"] != lock["test"]:
        raise SystemExit("deep.json test ids do not match the lock")
    complexes = load_cache()
    train = _take(complexes, lock["train"])
    val = _take(complexes, lock["val"])
    test = _take(complexes, lock["test"])
    runs = {}
    rows = {}
    for name in ("C", "Q"):
        runs[name] = []
        rows[name] = []
        for seed in (0, 1, 2):
            done = train_one(
                name, name, WIDTH, True, seed, train, val, test,
                build=build, steps=STEPS,
            )
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
    iface: list[list[float]] = []
    for row in paired:
        buckets.setdefault(row["cluster"], []).append(row["Q_minus_C_nll"])
        iface.append([row["Q_interface_nll"] - row["C_interface_nll"]])
    effect = cluster_interval(list(buckets.values()))
    print(
        f"Q-C nll {effect['point']:+.3f} [{effect['lo']:+.3f}, {effect['hi']:+.3f}] "
        f"clusters {effect['n_clusters']}",
        flush=True,
    )
    out = {
        "task": "800-step follow-up of the 3-block sum versus mean-plus-mass comparison",
        "label": "follow-up on the already examined deep test",
        "steps": STEPS,
        "prior_steps": 400,
        "depth": DEPTH,
        "h_dim": 64,
        "msg": 64,
        "hidden": WIDTH,
        "uncertainty": "95% cluster bootstrap, seeds averaged first; not retraining variation; not a new partition",
        "test_ids": lock["test"],
        "prior_effect": prior["effect"],
        "effect": effect,
        "interface_effect": cluster_interval(iface),
        "paired": paired,
        "runs": runs,
    }
    path = RESULTS / "deep_budget.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
