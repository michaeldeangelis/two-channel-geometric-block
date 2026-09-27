"""Stage 1: does the gated sum beat geometric softmax and softmax-plus-count?

Decision, fixed before looking at a full run:

- Train A, B, C, D and diagnostic R for three seeds.
- Score held-out MSE.
- D beats a baseline only when mean_D + std_D < mean_baseline - std_baseline.
- Pass: D beats A and D beats B.
- Kill: D does not beat B. Do not start Stage 2.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from block import TwoChannelBlock, parameter_count
from synthetic import H_DIM, sample_batch

ROOT = Path(__file__).resolve().parent
STEPS = 1200
BATCH = 128
LR = 2e-3
SEEDS = (0, 1, 2)
MSG_DIM = 16
BASE_HIDDEN = 64
VARIANTS = ("A", "B", "C", "D", "R")


def hidden_for_budget(target: int) -> dict[str, int]:
    """Widen the shared hidden size until each variant is within 10% of D.

    D owns both aggregators, so its hidden size stays at the base. The other
    variants spend the same parameter budget on a wider network.
    """
    widths = {"D": BASE_HIDDEN}
    for variant in VARIANTS:
        if variant == "D":
            continue
        hidden = BASE_HIDDEN
        count = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant))
        while count < int(0.90 * target) and hidden < 512:
            hidden += 8
            count = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant))
        widths[variant] = hidden
    return widths


def evaluate(model: TwoChannelBlock, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(batch["h"], batch["pos"], batch["mask"])
        return torch.mean((y_hat - batch["y"]) ** 2).item()


def train_one(variant: str, hidden: int, seed: int, val: dict[str, torch.Tensor]) -> dict[str, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    model = TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    last = math.inf
    history = []
    for step in range(1, STEPS + 1):
        batch = sample_batch(BATCH, gen)
        y_hat, _ = model(batch["h"], batch["pos"], batch["mask"])
        loss = torch.mean((y_hat - batch["y"]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            last = evaluate(model, val)
            history.append({"step": step, "val_mse": last})
            print(f"  {variant} seed {seed} step {step:4d}  val_mse {last:.4f}", flush=True)
            model.train()
    return {
        "val_mse": last,
        "params": float(parameter_count(model)),
        "hidden": float(hidden),
        "history": history,
    }


def beats(challenger: list[float], baseline: list[float]) -> bool:
    c_mean = sum(challenger) / len(challenger)
    b_mean = sum(baseline) / len(baseline)
    c_std = _std(challenger)
    b_std = _std(baseline)
    return (c_mean + c_std) < (b_mean - b_std)


def _std(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1))


def main() -> None:
    target = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, BASE_HIDDEN, "D"))
    widths = hidden_for_budget(target)
    print("parameter budget target", target)
    print("hidden widths", widths)
    val_gen = torch.Generator().manual_seed(12345)
    val = sample_batch(2000, val_gen)
    var_y = torch.mean(val["y"] ** 2).item()  # labels are mean-zero by symmetry
    print(f"zero-predictor mse {var_y:.4f}")

    runs: dict[str, list[dict[str, float]]] = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            runs[variant].append(train_one(variant, widths[variant], seed, val))

    summary = {}
    for variant, rows in runs.items():
        mses = [r["val_mse"] for r in rows]
        summary[variant] = {
            "mse_mean": sum(mses) / len(mses),
            "mse_std": _std(mses),
            "mse_seeds": mses,
            "params": rows[0]["params"],
            "hidden": rows[0]["hidden"],
            "skill": 1.0 - (sum(mses) / len(mses)) / var_y,
            "histories": [r["history"] for r in rows],
        }
        print(
            f"{variant}  mse {summary[variant]['mse_mean']:.4f} ± {summary[variant]['mse_std']:.4f}"
            f"  skill {summary[variant]['skill']:.3f}  params {int(rows[0]['params'])}"
        )

    d = [r["val_mse"] for r in runs["D"]]
    verdict = {
        "D_beats_A": beats(d, [r["val_mse"] for r in runs["A"]]),
        "D_beats_B": beats(d, [r["val_mse"] for r in runs["B"]]),
        "D_beats_C": beats(d, [r["val_mse"] for r in runs["C"]]),
        "R_beats_B": beats([r["val_mse"] for r in runs["R"]], [r["val_mse"] for r in runs["B"]]),
    }
    if verdict["D_beats_A"] and verdict["D_beats_B"]:
        decision = "pass"
    elif not verdict["D_beats_B"]:
        decision = "kill"
    else:
        decision = "inconclusive"
    print("verdict", verdict, "decision", decision)

    out = {
        "task": "half-space payload sum with smooth cutoff and signed payloads",
        "steps": STEPS,
        "seeds": list(SEEDS),
        "zero_predictor_mse": var_y,
        "summary": summary,
        "verdict": verdict,
        "decision": decision,
        "rule": "D beats baseline iff mean_D + std_D < mean_baseline - std_baseline",
    }
    path = ROOT / "results" / "stage1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
